

from typing import Union, Iterable
import torch
import torch.nn.functional as F


def cross_entropy(
    outs: torch.Tensor, 
    labels: torch.Tensor, 
    reduction: str = "mean"
) -> torch.Tensor:
    """
    cross entropy with logits
    """
    return F.cross_entropy(outs, labels, reduction=reduction)

def cross_entropy_softmax(
    probs: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    cross entropy with probs
        probs: the softmax of logits
    """
    return F.nll_loss(probs.log(), labels, reduction=reduction)

def kl_divergence(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    reduction: str = "batchmean"
) -> torch.Tensor:
    # KL divergence
    assert logits.size() == targets.size()
    # targets = targets.clone().detach()
    inputs = F.log_softmax(logits, dim=-1)
    targets = F.softmax(targets, dim=-1)
    return F.kl_div(inputs, targets, reduction=reduction)

def mse_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    return F.mse_loss(inputs, targets, reduction=reduction)

def lploss(
    x: torch.Tensor,
    p: Union[int, float, 'fro', 'nuc'] = 'fro',
    dim: Union[int, Iterable] = -1
):
    return torch.norm(x, p=p, dim=dim).mean()


def entropy_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    r = (inputs - targets).abs()
    # loss = (r * torch.log(r + 1e-6))
    loss = (r * torch.log(r + 1))
    if reduction == "mean":
        return loss.mean()
    else:
        return loss.sum()




class IdentityWrapper:

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"):
        r = (inputs - targets).abs()
        loss = r.pow(2)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)



class RefineWrapper(IdentityWrapper):

    def __init__(self, gamma: float = 1.) -> None:
        self.gamma = gamma

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"):
        r = (inputs - targets).abs()
        loss = - 1 / (r.pow(2) + 1).pow(self.gamma) + 1
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

class TanhWrapper(IdentityWrapper):

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"):
        r = (inputs - targets).abs()
        loss = r.pow(2).tanh()
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

class LogWrapper(IdentityWrapper):

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"):
        r = (inputs - targets).abs()
        loss = (r.pow(2) + 1).log()
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss


