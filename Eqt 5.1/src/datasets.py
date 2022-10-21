

from typing import Callable, Optional, Tuple, List, Any
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
from scipy.stats import uniform
import os
from PIL import Image

from .utils import getLogger
from .config import ROOT




class ShockWave(Dataset):

    def __init__(
        self, eps: float = 1e-2,
        bounds: Tuple[float] = (0, 1),
        num_samples: int = 1000,
        train=True
    ) -> None:
        super().__init__()

        self.eps = eps
        self.bounds = bounds
        self.num_samples = num_samples
        if train:
            x = uniform.rvs(loc=bounds[0], scale=bounds[1] - bounds[0], size=num_samples)
        else:
            x = np.linspace(bounds[0], bounds[1], num=num_samples)
        x = x.astype(np.float32)
        self.data = {
            'x': x,
            'y': self._u(x),
            'g1': self._u_grad_first_order(x),
            'g2': self._u_grad_second_order(x)
        }
        x = np.array([0, 1], dtype=np.float32)
        self.boundary = {
            'x': x,
            'y': self._u(x),
            'g1': self._u_grad_first_order(x),
            'g2': self._u_grad_second_order(x)
        }
        for key, value in self.boundary.items():
            self.boundary[key] = torch.tensor(value).unsqueeze(1)

    
    def _u(self, x):
        return np.cos(np.pi * x / 2) * (1  - np.exp(-2 * x / self.eps))
    
    def _u_grad_first_order(self, x):
        tmp = np.exp(-2 * x / self.eps)
        return -np.sin(np.pi * x / 2) * np.pi / 2 * (1 - tmp) \
                + np.cos(np.pi * x / 2) * tmp * 2 / self.eps

    def _u_grad_second_order(self, x):
        tmp = np.exp(-2 * x / self.eps)
        part1 = np.pi ** 2 / 4 * (1 - tmp) + tmp * 4 / (self.eps ** 2)
        part2 = 2 * np.pi / self.eps * tmp
        return -np.cos(np.pi / 2 * x) * part1 - np.sin(np.pi / 2 * x) * part2
    
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        index = [index]
        x = self.data['x'][index]
        y = self.data['y'][index]
        g1 = self.data['g1'][index]
        g2 = self.data['g2'][index]
        return x, y, g1, g2



class IdentityTransform:

    def __call__(self, x: Any) -> Any:
        return x

class OrderTransform:

    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def __call__(self, data: Tuple) -> List:
        return [transform(item) for item, transform in zip(data, self.transforms)]



class WrapperSet(Dataset):

    def __init__(
        self, dataset: Dataset,
        transforms: Optional[str] = None
    ) -> None:
        """
        Args:
            dataset: dataset;
            transforms: string spilt by ',', such as "tensor,none'
        """
        super().__init__()

        self.data = dataset

        try:
            counts = len(self.data[0])
        except IndexError:
            getLogger().info("[Dataset] zero-size dataset, skip ...")
            return

        if transforms is None:
            transforms = ['none'] * counts
        else:
            transforms = transforms.split(',')
        self.transforms = [AUGMENTATIONS[transform] for transform in transforms]
        if counts == 1:
            self.transforms = self.transforms[0]
        else:
            self.transforms = OrderTransform(self.transforms)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int):
        data = self.data[index]
        return self.transforms(data)


AUGMENTATIONS = {
    'none' : IdentityTransform(),
    'tensor': T.ToTensor(),
    'cifar': T.Compose((
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        )),
}

