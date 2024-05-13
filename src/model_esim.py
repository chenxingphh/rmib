import torch
import torch.nn as nn
import numpy as np
import math
from .modules.embedding import Embedding
from .modules.pooling import Pooling
from .modules import Module, ModuleList, ModuleDict

from .modules.prediction import registry as prediction, Linear

torch.autograd.set_detect_anomaly(True)


class AvgPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        # x: batch, seq_len, dim
        # mask: batch, seq_len, 1
        return torch.sum(x.masked_fill_(~mask.bool(), 0), dim=1) / torch.sum(mask.int(), dim=1)


class ESIM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.summary = {}

        self.embedding = Embedding(args)
        self.embed_dim = args.embedding_dim
        self.hidden_dim = args.embedding_dim // 2
        self.lstm_1 = nn.LSTM(args.embedding_dim, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True, )
        self.lstm_2 = nn.LSTM(4 * args.embedding_dim, self.hidden_dim, num_layers=1, bidirectional=True,
                              batch_first=True, )

        self.prediction = nn.Sequential(Linear(self.embed_dim * 4, self.embed_dim, activations=True),
                                        nn.Dropout(args.dropout),
                                        Linear(self.embed_dim, args.num_classes, ),  # activations=False
                                        )
        self.num_classes = args.num_classes
        self.dropout = nn.Dropout(args.dropout)

        self.fc_mean = nn.Sequential(
            nn.Linear(self.embed_dim * 4, self.embed_dim * 4),
            nn.ReLU())

        self.fc_std = nn.Sequential(
            nn.Linear(self.embed_dim * 4, self.embed_dim * 4),
            nn.Softplus())

        self.z_fc_mean = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim * 2),
            nn.ReLU())

        self.z_fc_std = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim * 2),
            nn.Softplus())

        self.ib_type = args.ib_type
        self.z_beat = args.z_beat
        self.kl_beta = args.kl_beta
        self.z_ce_loss_beat = args.z_ce_loss_beat

        self.z1_fc = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim * 4),
        )

        self.z2_fc = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim * 4),
            nn.ReLU(inplace=False)
        )

        self.fc_representation_discrimitor = nn.Sequential(
            nn.Linear(self.embed_dim * 4, self.embed_dim * 2),
            nn.ReLU(inplace=False),
            nn.Linear(self.embed_dim * 2, self.embed_dim * 2),
            nn.ReLU(inplace=False),
            nn.Linear(self.embed_dim * 2, 2),
            nn.Sigmoid()
        )

        self.cross_entropy = nn.BCELoss()
        self.pooling = Pooling()
        self.avg_pooling = AvgPooling()

    def forward(self, inputs):
        # x1: batch, max_len1
        # x1_mask: batch, max_len1
        # x2: batch, max_len2
        # x2_mask: batch, max_len2

        x1 = inputs['text1']
        x2 = inputs['text2']
        x1_mask = inputs['mask1']
        x2_mask = inputs['mask2']

        # x1: batch ,max_len1, dim
        # x2: batch, max_len2, dim
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        # Bi-LSTM
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x1, _ = self.lstm_1(x1)
        x2, _ = self.lstm_1(x2)

        x1 = self.dropout(x1.clone())
        x2 = self.dropout(x2.clone())
        x1_aligned = self.attn_align(x1, x2, x2_mask)
        x2_aligned = self.attn_align(x2, x1, x1_mask)

        # concat [x1, x1_aligned, x1-x1_aligned, x1 * x1_aligned]
        # x1_combined: batch, max_len1, dim
        # x2_combined: batch, max_len2, dim
        x1_combined = torch.cat([x1, x1_aligned, x1 - x1_aligned, x1 * x1_aligned], dim=-1)
        x2_combined = torch.cat([x2, x2_aligned, x2 - x2_aligned, x2 * x2_aligned], dim=-1)

        # Bi-LSTM
        # x1_: batch, max_len1, dim
        # x2_: batch, max_len2, dim
        x1_combined = self.dropout(x1_combined)
        x2_combined = self.dropout(x2_combined)
        x1, _ = self.lstm_2(x1_combined)
        x2, _ = self.lstm_2(x2_combined)

        a_max = self.pooling(x1.clone(), x1_mask)
        b_max = self.pooling(x2.clone(), x2_mask)

        a_avg = self.avg_pooling(x1.clone(), x1_mask)
        b_avg = self.avg_pooling(x2.clone(), x2_mask)

        x_final = torch.cat([a_avg, b_avg, a_max, b_max], dim=-1)

        if self.ib_type.lower() == 'none':
            logit = self.prediction(x_final)
            return logit, {}

        elif self.ib_type.lower() == 'rib':
            z1 = torch.cat([a_avg, a_max], dim=-1)
            z2 = torch.cat([b_avg, b_max], dim=-1)

            z1_mean, z1_std = self.z_fc_mean(z1), self.z_fc_std(z1)
            z2_mean, z2_std = self.z_fc_mean(z2), self.z_fc_std(z2)

            z1, z2 = self.reparametrize(z1_mean, z1_std, z2_mean, z2_std)
            Z = torch.cat([z1, z2], dim=-1)
            logit = self.prediction(Z)

            return logit, {'z1_mean': z1_mean, 'z1_std': z1_std, 'z2_mean': z2_mean, 'z2_std': z2_std}
        else:
            z1 = torch.cat([a_avg, a_max], dim=-1)
            z2 = torch.cat([b_avg, b_max], dim=-1)

            # I(z1;X), I(z2;x) => KL(p(z1|x)||q(x)), KL(p(z2|x)||q(x))
            z1_mean, z1_std = self.z_fc_mean(z1), self.z_fc_std(z1)
            z2_mean, z2_std = self.z_fc_mean(z2), self.z_fc_std(z2)

            z1, z2 = self.reparametrize(z1_mean, z1_std, z2_mean, z2_std)

            # I(Z;Y)
            Z = torch.cat([z1, z2], dim=-1)
            logit = self.prediction(Z)

            # I(z1;Y) and I(z2;Y)
            z1_final = self.z1_fc(z1)
            z2_final = self.z2_fc(z2)

            z1_logit = self.prediction(z1_final)
            z2_logit = self.prediction(z2_final)

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

    def attn_align(self, x, y, y_mask=None):
        # x: batch, max_len1, dim
        # y: batch, max_len2, dim
        # y_mask: batch, max_len2

        # attn_score: batch, max_len1, max_len2
        attn_score = x.matmul(y.transpose(1, 2))

        if y_mask != None:
            y_mask = y_mask.transpose(1, 2)
            attn_score = attn_score.masked_fill(y_mask.int() == 0, -1e9)

        # attn_score: batch, max_len1, max_len2
        attn_score = torch.softmax(attn_score, dim=1)

        # z: batch, max_len1, dim
        z = attn_score.matmul(y)

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
