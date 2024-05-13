import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraForPreTraining, AutoModel


class SentBert(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.summary = {}

        self.model_path = args.model_path
        self.model = AutoModel.from_pretrained(self.model_path)

        self.hidden_size = self.model.config.to_dict()['hidden_size']
        self.num_class = args.num_classes

        if 'quora' in args.data_dir and args.model == 'sbert':
            self.fc = nn.Sequential(
                nn.ReLU(),
                nn.Linear(3 * self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.num_class)
            )
        else:
            self.fc = nn.Linear(3 * self.hidden_size, self.num_class)

        self.ib_type = args.ib_type
        self.z_beat = args.z_beat
        self.kl_beta = args.kl_beta
        self.z_ce_loss_beat = args.z_ce_loss_beat

        self.z_fc_mean = nn.Sequential(
            nn.Linear(3 * self.hidden_size, 3 * self.hidden_size),
            nn.ReLU()
        )

        self.z_fc_std = nn.Sequential(
            nn.Linear(3 * self.hidden_size, 3 * self.hidden_size),
            nn.Softplus()
        )

        self.z_fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 3),
        )

        self.z_concat = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 3),
        )

        self.fc_representation_discrimitor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
            nn.Sigmoid()
        )

        self.cross_entropy = nn.BCELoss()

        self.fc_mean = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 3),
            nn.Tanh())

        self.fc_std = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 3),
            nn.Softplus())

    def forward(self, inputs):
        x1_input_ids = inputs['text1']
        x1_token_type_ids = inputs['text1_token_type_ids']
        x1_attention_mask = inputs['text1_attention_mask']
        x2_input_ids = inputs['text2']
        x2_token_type_ids = inputs['text2_token_type_ids']
        x2_attention_mask = inputs['text2_attention_mask']

        # x1_output: batch, len, dim
        x1_output = self.model(input_ids=x1_input_ids, token_type_ids=x1_token_type_ids,
                               attention_mask=x1_attention_mask).last_hidden_state

        # x1_output: batch,1,dim -> batch, dim
        x1_output = nn.AdaptiveMaxPool2d((1, self.hidden_size))(x1_output)
        x1_output = x1_output.squeeze(1)

        # x2_output: batch, len, dim
        x2_output = self.model(input_ids=x2_input_ids, token_type_ids=x2_token_type_ids,
                               attention_mask=x2_attention_mask).last_hidden_state
        # x2_output: batch,1,dim -> batch, dim
        x2_output = nn.AdaptiveMaxPool2d((1, self.hidden_size))(x2_output)
        x2_output = x2_output.squeeze(1)

        # x_combine:
        x_combine = torch.cat([x1_output, x2_output, torch.abs(x1_output - x2_output)], dim=-1)

        if self.ib_type.lower() == 'none':
            logit = self.fc(x_combine)
            return logit, {}

        elif self.ib_type.lower() == 'rib':
            z1 = x1_output
            z2 = x2_output

            # I(z1;X), I(z2;x) => KL(p(z1|x)||q(x)), KL(p(z2|x)||q(x))
            z1_mean, z1_std = self.z_fc_mean(z1), self.z_fc_std(z1)
            z2_mean, z2_std = self.z_fc_mean(z2), self.z_fc_std(z2)

            z1, z2 = self.reparametrize(z1_mean, z1_std)
            z = torch.cat([z1, z2], dim=-1)

            logit = self.fc(z)
            return logit, {'z1_mean': z1_mean, 'z1_std': z1_std, 'z2_mean': z2_mean, 'z2_std': z2_std}

        else:
            z1 = x1_output
            z2 = x2_output

            # I(Z;Y)
            z = torch.cat([z1, z2, torch.abs(z1 - z2)], dim=-1)

            # For SBert, we min I(z;X) instead of I(z1; X) and I(z2;X) because x1 and x2 don't interact before z
            z_mean, z_std = self.z_fc_mean(z), self.z_fc_std(z)
            z = self.reparametrize(z_mean, z_std)

            logit = self.fc(z)

            # I(z1;Y),I(z2;Y)
            z1_final = self.z_fc(z1)
            z2_final = self.z_fc(z2)

            z1_logit = self.fc(z1_final)
            z2_logit = self.fc(z2_final)

            z_loss = self.representation_discrimitor(z1, z2)

        return logit, {'z1_pred': z1_logit, 'z2_pred': z2_logit, 'z1_mean': z_mean, 'z1_std': z_std,
                       'z2_mean': z_mean, 'z2_std': z_std, 'z_loss': z_loss, 'num_class': self.num_class,
                       'z_beat': self.z_beat,
                       'kl_beta': self.kl_beta,
                       'z_ce_loss_beat': self.z_ce_loss_beat
                       }

    def representation_discrimitor(self, z1, z2):
        # z1: batch, dim
        # z2: batch, dim

        shuffle_idx = torch.randperm(z2.size(0))
        z2_shuffle = z2[shuffle_idx]

        pos = torch.cat([z1, z2], dim=-1)
        neg = torch.cat([z1, z2_shuffle], dim=-1)

        pos_pred = self.fc_representation_discrimitor(pos)
        neg_pred = self.fc_representation_discrimitor(neg)

        loss = -(torch.mean(pos_pred) - torch.log(torch.mean(torch.exp(neg_pred))))

        return loss

    def reparametrize(self, z_mean, z_std):
        z = torch.normal(0, 1, z_std.size(), requires_grad=False).to(z_std.device)
        z = z_mean + z * z_std

        return z

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
