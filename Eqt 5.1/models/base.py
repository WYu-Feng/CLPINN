

from typing import Any
import torch
import torch.nn as nn

from src.config import DEVICE


class EquArch(nn.Module):

    def __init__(
        self, model: nn.Module,
        device: torch.device = DEVICE
    ) -> None:
        super().__init__()
        self.arch = model
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(model)
        else:
            self.model = model.to(device)

    def state_dict(self, *args, **kwargs):
        return self.arch.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return self.arch.load_state_dict(*args, **kwargs)

    def __call__(self, inputs: torch.Tensor, **kwargs: Any) -> Any:
        return  self.model(inputs, **kwargs)

