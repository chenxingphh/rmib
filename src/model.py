import os
import math
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .network import Network
from .utils.metrics import registry as metrics
import torch.nn as nn
from .model_esim import ESIM
from .model_sbert import SentBert
from .model_bert import Bert


class LabelSmoothing(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def VIB_loss(y_pred, y_true, z1_mean, z1_std, z2_mean, z2_std, beta=0.1):
    #  compute I(z1;X) and I(z2;x) => KL(p(z1|x)||q(x)), KL(p(z2|x)||q(x))
    z1_kl_loss = 0.5 * torch.mean(
        torch.square(z1_mean) + torch.square(z1_std) - torch.log(1e-8 + torch.square(z1_std)) - 1)
    z2_kl_loss = 0.5 * torch.mean(
        torch.square(z2_mean) + torch.square(z2_std) - torch.log(1e-8 + torch.square(z2_std)) - 1)

    ce_loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)

    return beta * (z1_kl_loss + z2_kl_loss) + ce_loss


def RMIB_loss(y_pred, y_true,
              z1_pred, z2_pred,
              z1_mean, z1_std,
              z2_mean, z2_std,
              z_loss, num_class,
              z_beat=0.1,
              kl_beta=0.1,
              z_ce_loss_beat=0.1):
    #  compute I(z1;X) and I(z2;x) => KL(p(z1|x)||q(x)), KL(p(z2|x)||q(x))
    z1_kl_loss = 0.5 * torch.mean(
        torch.square(z1_mean) + torch.square(z1_std) - torch.log(1e-8 + torch.square(z1_std)) - 1)
    z2_kl_loss = 0.5 * torch.mean(
        torch.square(z2_mean) + torch.square(z2_std) - torch.log(1e-8 + torch.square(z2_std)) - 1)

    # compute I(Z;Y)
    ce_loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)

    # min I(z1; Y) and I(z2;Y)
    z1_ce_loss = LabelSmoothing(smoothing=1.0)(z1_pred, y_true)
    z2_ce_loss = LabelSmoothing(smoothing=1.0)(z2_pred, y_true)

    # I(z1;z2)->z_loss
    return ce_loss + z_beat * z_loss + kl_beta * (z1_kl_loss + z2_kl_loss) + z_ce_loss_beat * (z1_ce_loss + z2_ce_loss)


class Model:
    prefix = 'checkpoint'
    best_model_name = 'best.pt'

    def __init__(self, args, state_dict=None):
        self.args = args

        if args.model == 're2':
            self.network = Network(args)
        elif args.model == 'esim':
            self.network = ESIM(args)
        elif args.model == 'sbert':
            self.network = SentBert(args)
        elif args.model == 'bert':
            self.network = Bert(args)
        else:
            raise NotImplementedError

        self.device = torch.cuda.current_device() if args.cuda else torch.device('cpu')
        self.network.to(self.device)

        self.params = list(filter(lambda x: x.requires_grad, self.network.parameters()))
        self.opt = torch.optim.Adam(self.params, args.lr, betas=(args.beta1, args.beta2),
                                    weight_decay=args.weight_decay)
        # updates
        self.updates = state_dict['updates'] if state_dict else 0

        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['model'].keys()):
                if k not in new_state:
                    del state_dict['model'][k]
            self.network.load_state_dict(state_dict['model'])
            self.opt.load_state_dict(state_dict['opt'])

    def _update_schedule(self):
        if self.args.lr_decay_rate < 1.:
            args = self.args
            t = self.updates
            base_ratio = args.min_lr / args.lr
            if t < args.lr_warmup_steps:
                ratio = base_ratio + (1. - base_ratio) / max(1., args.lr_warmup_steps) * t
            else:
                ratio = max(base_ratio, args.lr_decay_rate ** math.floor((t - args.lr_warmup_steps) /
                                                                         args.lr_decay_steps))
            self.opt.param_groups[0]['lr'] = args.lr * ratio

    def update(self, batch):
        self.network.train()
        self.opt.zero_grad()

        inputs, target = self.process_data(batch)
        output, res = self.network(inputs)
        summary = self.network.get_summary()

        loss = self.get_loss(output, target, self.network.ib_type, **res)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_clipping)

        assert grad_norm >= 0, 'encounter nan in gradients.'
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()

        self.opt.step()
        self._update_schedule()
        self.updates += 1

        stats = {
            'updates': self.updates,
            'loss': loss.item(),
            'lr': self.opt.param_groups[0]['lr'],
            'gnorm': grad_norm,
            'summary': summary,
        }
        return stats

    def evaluate(self, data):
        self.network.eval()
        targets = []
        probabilities = []
        predictions = []
        losses = []

        for batch in tqdm(data[:self.args.eval_subset], desc='evaluating', leave=False):
            inputs, target = self.process_data(batch)
            with torch.no_grad():
                output, res = self.network(inputs)
                loss = self.get_loss(output, target, self.network.ib_type, **res)
                pred = torch.argmax(output, dim=1)
                prob = torch.nn.functional.softmax(output, dim=1)
                losses.append(loss.item())
                targets.extend(target.tolist())
                probabilities.extend(prob.tolist())
                predictions.extend(pred.tolist())
        outputs = {
            'target': targets,
            'prob': probabilities,
            'pred': predictions,
            'args': self.args,
        }
        stats = {
            'updates': self.updates,
            'loss': sum(losses[:-1]) / (len(losses) - 1) if len(losses) > 1 else sum(losses),
        }
        for metric in self.args.watch_metrics:
            if metric not in stats:  # multiple metrics could be computed by the same function
                stats.update(metrics[metric](outputs))
        assert 'score' not in stats, 'metric name collides with "score"'
        eval_score = stats[self.args.metric]
        stats['score'] = eval_score
        return eval_score, stats  # first value is for early stopping

    def predict(self, batch):
        self.network.eval()
        inputs, _ = self.process_data(batch)
        with torch.no_grad():
            output, _ = self.network(inputs)
            output = torch.nn.functional.softmax(output, dim=1)
        return output.tolist()

    def process_data(self, batch):
        if self.args.model not in ['sbert', 'bert']:
            text1 = torch.LongTensor(batch['text1']).to(self.device)
            text2 = torch.LongTensor(batch['text2']).to(self.device)
            mask1 = torch.ne(text1, self.args.padding).to(dtype=torch.bool).unsqueeze(2)
            mask2 = torch.ne(text2, self.args.padding).to(dtype=torch.bool).unsqueeze(2)
            inputs = {
                'text1': text1,
                'text2': text2,
                'mask1': mask1,
                'mask2': mask2,
            }
            if 'target' in batch:
                target = torch.LongTensor(batch['target']).to(self.device)
                return inputs, target
            return inputs, None
        elif self.args.model == 'sbert':
            if isinstance(batch, list):
                batch = {key: [sample[key] for sample in batch] for key in batch[0].keys()}

            text1 = torch.LongTensor(batch['text1']).to(self.device)
            text2 = torch.LongTensor(batch['text2']).to(self.device)
            text1_token_type_ids = torch.LongTensor(batch['text1_token_type_ids']).to(self.device)
            text2_token_type_ids = torch.LongTensor(batch['text2_token_type_ids']).to(self.device)
            text1_attention_mask = torch.LongTensor(batch['text1_attention_mask']).to(self.device)
            text2_attention_mask = torch.LongTensor(batch['text2_attention_mask']).to(self.device)

            inputs = {
                'text1': text1,
                'text2': text2,
                'text1_token_type_ids': text1_token_type_ids,
                'text2_token_type_ids': text2_token_type_ids,
                'text1_attention_mask': text1_attention_mask,
                'text2_attention_mask': text2_attention_mask,
            }
            if 'target' in batch:
                target = torch.LongTensor(batch['target']).to(self.device)
                return inputs, target
            return inputs, None
        elif self.args.model == 'bert':
            if isinstance(batch, list):
                batch = {key: [sample[key] for sample in batch] for key in batch[0].keys()}

            text = torch.LongTensor(batch['text']).to(self.device)
            text_token_type_ids = torch.LongTensor(batch['text_token_type_ids']).to(self.device)
            text_attention_mask = torch.LongTensor(batch['text_attention_mask']).to(self.device)
            text1_mask = torch.LongTensor(batch['text1_mask']).to(self.device)
            text2_mask = torch.LongTensor(batch['text2_mask']).to(self.device)

            inputs = {
                'text': text,
                'text_token_type_ids': text_token_type_ids,
                'text_attention_mask': text_attention_mask,
                'text1_mask': text1_mask,
                'text2_mask': text2_mask,
            }
            if 'target' in batch:
                target = torch.LongTensor(batch['target']).to(self.device)
                return inputs, target
            return inputs, None

        else:
            raise NotImplementedError

    @staticmethod
    def get_loss(logits, target, ib_type='None', **res):

        if ib_type.lower() == 'none':
            return F.cross_entropy(logits, target)
        elif ib_type.lower() == 'rib':
            return VIB_loss(logits, target, **res)
        else:
            return RMIB_loss(logits, target, **res)

    def save(self, states, name=None):
        if name:
            filename = os.path.join(self.args.summary_dir, name)
        else:
            filename = os.path.join(self.args.summary_dir, f'{self.prefix}_{self.updates}.pt')
        params = {
            'state_dict': {
                'model': self.network.state_dict(),
                'opt': self.opt.state_dict(),
                'updates': self.updates,
            },
            'args': self.args,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state()
        }
        params.update(states)
        if self.args.cuda:
            params['torch_cuda_state'] = torch.cuda.get_rng_state()
        torch.save(params, filename)

    @classmethod
    def load(cls, file):
        checkpoint = torch.load(file, map_location=(
            lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
        ))
        prev_args = checkpoint['args']
        # update args
        prev_args.output_dir = os.path.dirname(os.path.dirname(file))
        prev_args.summary_dir = os.path.join(prev_args.output_dir, prev_args.name)
        prev_args.cuda = prev_args.cuda and torch.cuda.is_available()
        return cls(prev_args, state_dict=checkpoint['state_dict']), checkpoint

    def num_parameters(self, exclude_embed=False):
        num_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        if exclude_embed:
            num_params -= 0 if self.args.fix_embeddings else next(self.network.embedding.parameters()).numel()
        return num_params

    def set_embeddings(self, embeddings):
        self.network.embedding.set_(embeddings)
