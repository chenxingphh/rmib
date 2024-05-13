import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraForPreTraining, AutoModel
from transformers import AutoTokenizer
import numpy as np


class Bert(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.summary = {}

        self.model_path = args.model_path
        self.model = AutoModel.from_pretrained(self.model_path)
        self.hidden_size = self.model.config.to_dict()['hidden_size']
        self.num_classes = args.num_classes

        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.ib_type = args.ib_type

        self.z_beat = args.z_beat
        self.kl_beta = args.kl_beta
        self.z_ce_loss_beat = args.z_ce_loss_beat

        self.z_fc_mean = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )

        self.z_fc_std = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Softplus()
        )
        self.z_fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.z2_fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 3),
            nn.ReLU()
        )
        self.fc_representation_discrimitor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, 1),
            nn.Sigmoid()
        )

        self.prediction = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def forward(self, inputs):
        input_ids = inputs['text']
        token_type_ids = inputs['text_token_type_ids']
        attention_mask = inputs['text_attention_mask']
        text1_mask = inputs['text1_mask']
        text2_mask = inputs['text2_mask']

        # x1_output: batch, len, dim
        output = self.model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask).last_hidden_state

        # output: batch,1,dim -> batch, dim
        output_pool = nn.AdaptiveMaxPool2d((1, self.hidden_size))(output)
        output_pool = output_pool.squeeze(1)

        a = nn.AdaptiveMaxPool2d((1, self.hidden_size))(output * text1_mask.unsqueeze(-1))
        a = a.squeeze(1)

        b = nn.AdaptiveMaxPool2d((1, self.hidden_size))(output * text2_mask.unsqueeze(-1))
        b = b.squeeze(1)

        if self.ib_type.lower() == 'none':
            logit = self.prediction(output_pool)
            return logit, {}

        elif self.ib_type.lower() == 'rib':
            z1_, z2_ = a, b

            z1_mean, z1_std = self.z_fc_mean(z1_), self.z_fc_std(z1_)
            z2_mean, z2_std = self.z_fc_mean(z2_), self.z_fc_std(z2_)

            z1, z2 = self.reparametrize(z1_mean, z1_std, z2_mean, z2_std)
            z = self.fc(torch.cat([z1, z2], dim=-1))

            logit = self.prediction(z)

            return logit, {'z1_mean': z1_mean, 'z1_std': z1_std, 'z2_mean': z2_mean, 'z2_std': z2_std}
        else:
            z1, z2 = a, b

            # KL(p(z1|x)||q(x)), KL(p(z2|x)||q(x))
            z1_mean, z1_std = self.z_fc_mean(z1), self.z_fc_std(z1)
            z2_mean, z2_std = self.z_fc_mean(z2), self.z_fc_std(z2)

            # I(z1,z2;Y)
            z1, z2 = self.reparametrize(z1_mean, z1_std, z2_mean, z2_std)
            z = self.fc(torch.cat([z1, z2], dim=-1))
            logit = self.prediction(z)

            # I(z1;Y),I(z2;Y)
            z1_logit = self.prediction(self.z_fc(z1))
            z2_logit = self.prediction(self.z_fc(z2))

            # I(z1;z2)
            z_loss = self.representation_discrimitor(z1, z2)

            return logit, {'z1_pred': z1_logit, 'z2_pred': z2_logit, 'z1_mean': z1_mean, 'z1_std': z1_std,
                           'z2_mean': z2_mean, 'z2_std': z2_std, 'z_loss': z_loss, 'num_class': self.num_classes,
                           'z_beat': self.z_beat,
                           'kl_beta': self.kl_beta,
                           'z_ce_loss_beat': self.z_ce_loss_beat
                           }

    def representation_discrimitor(self, z1, z2):
        # z1: batch, dim
        # z2: batch, dim

        # shuffle the first dimension
        shuffle_idx = torch.randperm(z2.size(0))
        z2_shuffle = z2[shuffle_idx]

        pos = torch.cat([z1, z2], dim=-1)
        neg = torch.cat([z1, z2_shuffle], dim=-1)

        pos_pred = self.fc_representation_discrimitor(pos)
        neg_pred = self.fc_representation_discrimitor(neg)

        loss = -(torch.mean(pos_pred) - torch.log(torch.mean(torch.exp(neg_pred))))

        return loss

    def reparametrize(self, z1_mean, z1_std, z2_mean, z2_std):
        z = torch.normal(0, 1, z1_std.size(), requires_grad=False).to(z1_std.device)
        z_ = torch.normal(0, 1, z1_std.size(), requires_grad=False).to(z1_std.device)

        z1 = z1_mean + z * z1_std
        z2 = z2_mean + z_ * z2_std

        return z1, z2

    def add_summary(self, name, val):
        if self.training:
            self.summary[name] = val.clone().detach().cpu().numpy()

    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        if self.summary:
            summary.update({base_name + name: val for name, val in self.summary.items()})
        for name, child in self.named_children():
            if hasattr(child, 'get_summary'):
                name = base_name + name
                summary.update(child.get_summary(name))
        return summary
