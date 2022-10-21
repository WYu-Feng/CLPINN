

import torch
import torch.nn as nn
from .layerops import Linear


class LinearNet(nn.Module):

    def __init__(
        self, 
        in_features: int = 1,
        latent_features: int = 20,
    ) -> None:
        super().__init__()

       
        self.act = nn.Tanh()
        self.dense = nn.Sequential(
            Linear(in_features, latent_features),
            self.act,
            Linear(latent_features, latent_features),
            self.act,
        )
        self.linear = Linear(latent_features, 1)
        nn.init.uniform_(self.linear.weight, -0.5, 0.5)
        nn.init.uniform_(self.linear.bias, -0.5, 0.5)

    def forward(self, x):
        return self.linear(self.dense(x))

