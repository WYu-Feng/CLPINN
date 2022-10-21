
from typing import Callable, Tuple
import torch

import time
from tqdm import tqdm


from .config import *
from .utils import getLogger, mkdirs



class ModelNotDefineError(Exception): pass
class LossNotDefineError(Exception): pass
class OptimNotIncludeError(Exception): pass
class AttackNotIncludeError(Exception): pass
class DatasetNotIncludeError(Exception): pass



def load_model(model_type: str) -> Callable[..., torch.nn.Module]:
    if model_type == "linear":
        from models.LinearNet import LinearNet
        model = LinearNet
    elif model_type == "exp":
        from models.ExpNet import ExpNet
        model = ExpNet
    else:
        raise ModelNotDefineError(f"model {model_type} is not defined.\n" \
                f"Refer to the following: {load_model.__doc__}\n")
    return model

def load_loss_func(loss_type: str) -> Callable:
    """
    cross_entropy: the cross entropy loss with logits
    cross_entropy_softmax: the cross entropy loss with probs
    kl_loss: kl divergence
    mse_loss: MSE
    """
    loss_func: Callable[..., torch.Tensor]
    if loss_type == "cross_entropy":
        from .loss_zoo import cross_entropy
        loss_func = cross_entropy
    elif loss_type == "mse_loss":
        from .loss_zoo import mse_loss
        loss_func = mse_loss
    elif loss_type == "identity":
        from .loss_zoo import IdentityWrapper
        loss_func = IdentityWrapper()
    elif loss_type == "tanh":
        from .loss_zoo import TanhWrapper
        loss_func = TanhWrapper()
    elif loss_type == "log":
        from .loss_zoo import LogWrapper
        loss_func = LogWrapper()
    elif loss_type == "refine":
        from .loss_zoo import RefineWrapper
        loss_func = RefineWrapper()
    else:
        raise LossNotDefineError(f"Loss {loss_type} is not defined.\n" \
                    f"Refer to the following: {load_loss_func.__doc__}")
    return loss_func


def _dataset(
    dataset_type: str,
    eps: float = 1e-3,
    bounds: Tuple[float] = (0., 1.),
    num_samples: int = 1000,
    train: bool = True
):
    from .datasets import ShockWave
    if dataset_type == "shockwave":
        return ShockWave(eps, bounds, num_samples, train=train)
    else:
        raise DatasetNotIncludeError("No such dataset ...")


def load_dataset(
    dataset_type: str, 
    transforms: str ='default', 
    eps: float = 1e-3,
    bounds: Tuple[float] = (0., 1.),
    num_samples: int = 1000,
    train: bool = True,
) -> torch.utils.data.Dataset:
    from .datasets import WrapperSet, ShockWave
    dataset = _dataset(dataset_type, eps, bounds, num_samples, train=train)
    transforms = TRANSFORMS
    getLogger().info(f"[Dataset] Apply transforms of '{transforms}' to dataset ...")
    dataset = WrapperSet(dataset, transforms=transforms)
    return dataset


class _TQDMDataLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        return iter(
            tqdm(
                super(_TQDMDataLoader, self).__iter__(), 
                leave=False, desc="վ'ᴗ' ի-"
            )
        )

def load_dataloader(
    dataset: torch.utils.data.Dataset, 
    batch_size: int, 
    train: bool = True, 
    show_progress: bool = False
) -> torch.utils.data.DataLoader:

    dataloader = _TQDMDataLoader if show_progress else torch.utils.data.DataLoader
    if train:
        loader = dataloader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
    else:
        loader = dataloader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
    return loader


def load_optimizer(
    model: torch.nn.Module, 
    optim_type: str, *,
    lr: float = 0.1, momentum: float = 0.9,
    betas: Tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 1e-4,
    nesterov: bool = False,
    **kwargs: "other hyper-parameters for optimizer"
) -> torch.optim.Optimizer:
    """
    sgd: SGD
    adam: Adam
    """
    try:
        cfg = OPTIMS[optim_type]
    except KeyError:
        raise OptimNotIncludeError(f"Optim {optim_type} is not included.\n" \
                        f"Refer to the following: {load_optimizer.__doc__}")
    
    kwargs.update(lr=lr, momentum=momentum, betas=betas, 
                weight_decay=weight_decay, nesterov=nesterov)
    
    cfg.update(**kwargs) # update the kwargs needed automatically
    logger = getLogger()
    logger.info(cfg)
    if optim_type == "sgd":
        optim = torch.optim.SGD(model.parameters(), **cfg)
    elif optim_type == "adam":
        optim = torch.optim.Adam(model.parameters(), **cfg)
    elif optim_type == "lbfgs":
        optim = torch.optim.LBFGS(model.parameters(), **cfg)

    return optim


def load_learning_policy(
    optimizer: torch.optim.Optimizer,
    learning_policy_type: str,
    **kwargs: "other hyper-parameters for learning scheduler"
) -> "learning policy":
    """
    default: (100, 105), 110 epochs suggested
    null:
    STD: (82, 123), 164 epochs suggested
    STD-wrn: (60, 120, 160), 200 epochs suggested
    AT: (102, 154), 200 epochs suggested
    TRADES: (75, 90, 100), 76 epochs suggested
    TRADES-M: (55, 75, 90), 100 epochs suggested
    cosine: CosineAnnealingLR, kwargs: T_max, eta_min, last_epoch
    """
    try:
        learning_policy_ = LEARNING_POLICY[learning_policy_type]
    except KeyError:
        raise NotImplementedError(f"Learning_policy {learning_policy_type} is not defined.\n" \
            f"Refer to the following: {load_learning_policy.__doc__}")

    lp_type = learning_policy_[0]
    lp_cfg = learning_policy_[1]
    lp_cfg.update(**kwargs) # update the kwargs needed automatically
    logger = getLogger()
    logger.info(f"{lp_cfg}    {lp_type}")
    learning_policy = getattr(
        torch.optim.lr_scheduler, 
        lp_type
    )(optimizer, **lp_cfg)
    
    return learning_policy


def generate_path(
    method: str, dataset_type: str, model:str, description: str
) -> Tuple[str, str]:
    info_path = INFO_PATH.format(
        method=method,
        dataset=dataset_type,
        model=model,
        description=description
    )
    log_path = LOG_PATH.format(
        method=method,
        dataset=dataset_type,
        model=model,
        description=description,
        time=time.strftime(TIMEFMT)
    )
    mkdirs(info_path, log_path)
    return info_path, log_path

