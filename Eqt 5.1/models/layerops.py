
import torch
import torch.nn as nn
import torch.nn.functional as F


import math


def active_init(weight, bias, low: float = 0., high: float = 1.):
    assert high > low, "high should be greater than low"
    out_channels, in_channels = weight.size()
    D = math.sqrt(in_channels * (high - low))
    w = 8.72 * math.sqrt(3 / in_channels) / D
    nn.init.uniform_(weight, -w, w)
    C = -torch.ones(in_channels) * ((high - low) / 2)
    nn.init.constant_(bias, weight.data[0] @ C)

def nguyen_widrow_init(weight, bias, scale: float = 1.):
    out_channels, in_channels = weight.size()
    nn.init.uniform_(weight, -0.5, 0.5)
    scale = scale * out_channels ** (1 / in_channels)
    weight.data.copy_(scale * F.normalize(weight.data, p=1, dim=-1))
    nn.init.uniform_(bias, -scale, scale)


class Linear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = ...) -> None:
        super().__init__(in_features, out_features, bias)
        nguyen_widrow_init(self.weight, self.bias)
