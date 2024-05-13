import torch
from .modules import Module, ModuleList, ModuleDict
from .modules.embedding import Embedding
from .modules.encoder import Encoder
from .modules.alignment import registry as alignment
from .modules.fusion import registry as fusion
from .modules.connection import registry as connection
from .modules.pooling import Pooling
from .modules.prediction import registry as prediction, Linear
import torch.nn as nn
import numpy as np


class AvgPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        # x: batch, seq_len, dim
        # mask: batch, seq_len, 1
        return torch.sum(x.masked_fill_(~mask.bool(), 0), dim=1) / torch.sum(mask.int(), dim=1)


class Network(Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding = Embedding(args)

        input_emb_size = args.embedding_dim if args.connection == 'aug' else 0
        self.blocks = ModuleList([ModuleDict({
            'encoder': Encoder(args, args.embedding_dim if i == 0 else input_emb_size + args.hidden_size),
            'alignment': alignment[args.alignment](
                args, args.embedding_dim + args.hidden_size if i == 0 else input_emb_size + args.hidden_size * 2),
            'fusion': fusion[args.fusion](
                args, args.embedding_dim + args.hidden_size if i == 0 else input_emb_size + args.hidden_size * 2),
        }) for i in range(args.blocks)])

        self.connection = connection[args.connection](args)
        self.pooling = Pooling()
        self.prediction = prediction[args.prediction](args)

        self.z_fc_mean = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU()
        )

        self.z_fc_std = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.Softplus()

        )

        self.z1_fc = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size * 2)
        )

        self.z2_fc = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size * 2)
        )

        self.z_beat = args.z_beat
        self.kl_beta = args.kl_beta
        self.z_ce_loss_beat = args.z_ce_loss_beat

        self.num_classes = args.num_classes

        self.fc_representation_discrimitor = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(args.hidden_size * 2, 1),
            nn.Sigmoid()
        )

        self.cross_entropy = nn.BCELoss()
        self.ib_type = args.ib_type

    def forward(self, inputs):
        a = inputs['text1']
        b = inputs['text2']
        mask_a = inputs['mask1']
        mask_b = inputs['mask2']

        a = self.embedding(a)
        b = self.embedding(b)
        res_a, res_b = a, b

        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                b = self.connection(b, res_b, i)
                res_a, res_b = a, b
            a_enc = block['encoder'](a, mask_a)
            b_enc = block['encoder'](b, mask_b)
            a = torch.cat([a, a_enc], dim=-1)
            b = torch.cat([b, b_enc], dim=-1)
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)

        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)

        if self.ib_type.lower() == 'none':
            logit = self.prediction(a, b)
            return logit, {}

        elif self.ib_type.lower() == 'rib':
            z1_, z2_ = a, b

            z1_mean, z1_std = self.z_fc_mean(z1_), self.z_fc_std(z1_)
            z2_mean, z2_std = self.z_fc_mean(z2_), self.z_fc_std(z2_)

            z1, z2 = self.reparametrize(z1_mean, z1_std, z2_mean, z2_std)

            logit = self.prediction(z1, z2)

            return logit, {'z1_mean': z1_mean, 'z1_std': z1_std, 'z2_mean': z2_mean, 'z2_std': z2_std}
        else:
            z1, z2 = a, b
            z1_mean, z1_std = self.z_fc_mean(z1), self.z_fc_std(z1)
            z2_mean, z2_std = self.z_fc_mean(z2), self.z_fc_std(z2)

            z1, z2 = self.reparametrize(z1_mean, z1_std, z2_mean, z2_std)
            logit = self.prediction(z1, z2)

            z1_final = self.z1_fc(z1)
            z2_final = self.z2_fc(z2)

            z1_final_1, z1_final_2 = torch.chunk(z1_final, dim=-1, chunks=2)
            z2_final_1, z2_final_2 = torch.chunk(z2_final, dim=-1, chunks=2)

            z1_logit = self.prediction(z1_final_1, z1_final_2)
            z2_logit = self.prediction(z2_final_1, z2_final_2)

            z_loss = self.representation_discrimitor(z1, z2)

            return logit, {'z1_pred': z1_logit, 'z2_pred': z2_logit, 'z1_mean': z1_mean, 'z1_std': z1_std,
                           'z2_mean': z2_mean, 'z2_std': z2_std, 'z_loss': z_loss, 'num_class': self.num_classes,
                           'z_beat': self.z_beat,
                           'kl_beta': self.kl_beta,
                           'z_ce_loss_beat': self.z_ce_loss_beat,
                           }

    def reparametrize(self, z1_mean, z1_std, z2_mean, z2_std):
        z = torch.normal(0, 1, z1_std.size(), requires_grad=False).to(z1_std.device)
        z_ = torch.normal(0, 1, z1_std.size(), requires_grad=False).to(z1_std.device)

        z1 = z1_mean + z * z1_std
        z2 = z2_mean + z_ * z2_std

        return z1, z2

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
