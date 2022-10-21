


import torch
import torch.nn as nn
from .layerops import ExpLinear, Sin


class ExpNet(nn.Module):

    def __init__(
        self, 
        in_features: int = 1,
        latent_features: int = 20,
        activation: str = 'sin'
    ) -> None:
        super().__init__()

        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU(True)
        elif activation == "sin":
            self.act = Sin()
        else:
            raise ValueError("No such activation ...")
        
        self.dense = nn.Sequential(
            ExpLinear(in_features, latent_features),
            self.act,
            ExpLinear(latent_features, latent_features),
            self.act,
            nn.Linear(latent_features, 1)
        )

    def forward(self, x):
        return self.dense(x)

