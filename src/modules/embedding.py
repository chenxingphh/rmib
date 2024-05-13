import torch
import torch.nn as nn
import torch.nn.functional as f


class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fix_embeddings = args.fix_embeddings
        self.embedding = nn.Embedding(args.num_vocab, args.embedding_dim, padding_idx=0)
        self.dropout = args.dropout

    def set_(self, value):
        self.embedding.weight.requires_grad = not self.fix_embeddings
        self.embedding.load_state_dict({'weight': torch.tensor(value)})

    def forward(self, x):
        x = self.embedding(x)
        x = f.dropout(x, self.dropout, self.training)
        return x
