import math
import torch
import torch.nn as nn
from . import Linear
from functools import partial
from src.utils.registry import register
registry = {}
register = partial(register, registry=registry)


@register('none')
class NullConnection(nn.Module):
    def __init__(self, _):
        super().__init__()

    def forward(self, x, _, __):
        return x


@register('residual')
class Residual(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear = Linear(args.embedding_dim, args.hidden_size)

    def forward(self, x, res, i):
        if i == 1:
            res = self.linear(res)
        return (x + res) * math.sqrt(0.5)


@register('aug')
class AugmentedResidual(nn.Module):
    def __init__(self, _):
        super().__init__()

    def forward(self, x, res, i):
        if i == 1:
            return torch.cat([x, res], dim=-1)  # res is embedding
        hidden_size = x.size(-1)
        x = (res[:, :, :hidden_size] + x) * math.sqrt(0.5)
        return torch.cat([x, res[:, :, hidden_size:]], dim=-1)  # latter half of res is embedding
