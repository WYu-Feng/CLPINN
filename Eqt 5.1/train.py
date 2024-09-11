#!/usr/bin/env python


from typing import Tuple
import argparse

from src.loadopts import *
from src.utils import timemeter
from src.config import DEVICE



METHOD = "Equation"
SAVE_FREQ = 5
FMT = "{description}={leverage}-{eps}-{nums}" \
        "={learning_policy}-{optimizer}-{lr}" \
        "={epochs}" \
        "={batch_size}={transform}"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='linear')
parser.add_argument("--dataset", type=str, default='shockwave')
parser.add_argument("--info_path", type=str, default=None)

# for the shockwave dataset
parser.add_argument("--eps", type=float, default=1e-9)
parser.add_argument("--nums", type=int, default=2500)
parser.add_argument("--nums-valid", type=int ,default=1000)

# for expnet
parser.add_argument("--dim-latent", type=int, default=20)

# for regularization
parser.add_argument("--leverage", type=float, default=1.)

# basic settings
parser.add_argument("--loss", type=str, default="identity", choices=("identity", "tanh", "log", "refine"))
parser.add_argument("--optimizer", type=str, choices=("sgd", "adam"), default="adam")
parser.add_argument("-mom", "--momentum", type=float, default=0.9,
                help="the momentum used for SGD")
parser.add_argument("-beta1", "--beta1", type=float, default=0.9,
                help="the first beta argument for Adam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                help="the second beta argument for Adam")
parser.add_argument("-wd", "--weight_decay", type=float, default=0.,
                help="weight decay")
parser.add_argument("-lr", "--lr", "--LR", "--learning_rate", type=float, default=0.001)
parser.add_argument("-lp", "--learning_policy", type=str, default="null", 
                help="learning rate schedule defined in config.py")
parser.add_argument("--epochs", type=int ,default=3000)
parser.add_argument("-b", "--batch_size", type=int, default=50)
parser.add_argument("--G", type=int, default=10)
parser.add_argument("--transform", type=str, default='default', 
                help="the data augmentations which will be applied during training.")


# eval
parser.add_argument("--eval-train", action="store_true", default=False)
parser.add_argument("--eval-valid", action="store_false", default=True)
parser.add_argument("--eval-freq", type=int, default=100,
                help="for valid dataset only")

parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--progress", action="store_true", default=False, 
                help="show the progress if true")
parser.add_argument("--log2file", action="store_false", default=True,
                help="False: remove file handler")
parser.add_argument("--log2console", action="store_false", default=True,
                help="False: remove console handler if log2file is True ...")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--benchmark", action="store_true", default=False)
parser.add_argument("-m", "--description", type=str, default=METHOD)

opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)



def oracle(x, g1, g2, eps=opts.eps):
    return -eps * g2 - (2 - x) * g1

@timemeter("Setup")
def load_cfg() -> Tuple[Config, str]:
    from src.dict2obj import Config
    from src.base import Coach
    from src.utils import set_seed, activate_benchmark, load_checkpoint, set_logger, load
    from models.base import EquArch

    cfg = Config()
    
    # generate the path for logging information and saving parameters
    cfg['info_path'], cfg['log_path'] = generate_path(
        method=METHOD, dataset_type=opts.dataset, 
        model=opts.model, description=opts.description
    )
    # set logger
    logger = set_logger(
        path=cfg.log_path, 
        log2file=opts.log2file, 
        log2console=opts.log2console
    )

    activate_benchmark(opts.benchmark)
    set_seed(opts.seed)

    # the model and other settings for training
    model = load_model(opts.model)(
        latent_features=opts.dim_latent
    )
    model = EquArch(model=model)
    
    if opts.info_path is not None:
        load(
            model, opts.info_path
        )

    # load the dataset
    trainset = load_dataset(
        dataset_type=opts.dataset,
        transforms=opts.transform,
        eps=opts.eps,
        num_samples=opts.nums,
        train=True
    )
    validset = load_dataset(
        dataset_type=opts.dataset,
        transforms="tensor,none",
        eps=opts.eps,
        num_samples=opts.nums_valid,
        train=False
    )
    cfg['trainloader'] = load_dataloader(
        dataset=trainset,
        batch_size=opts.batch_size,
        train=True,
        show_progress=opts.progress
    )
    cfg['validloader'] = load_dataloader(
        dataset=validset,
        batch_size=opts.nums_valid,
        train=False,
        show_progress=opts.progress
    )
    cfg['boundary'] = trainset.data.boundary

    # load the optimizer and learning_policy
    optimizer = load_optimizer(
        model=model, optim_type=opts.optimizer, lr=opts.lr,
        momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
        weight_decay=opts.weight_decay
    )
    learning_policy = load_learning_policy(
        optimizer=optimizer, 
        learning_policy_type=opts.learning_policy,
        T_max=opts.epochs
    )

    if opts.resume:
        cfg['start_epoch'] = load_checkpoint(
            path=cfg.info_path, model=model, 
            optimizer=optimizer, 
            lr_scheduler=learning_policy
        )
    else:
        cfg['start_epoch'] = 0

    cfg['coach'] = Coach(
        model=model,
        loss_func=load_loss_func(opts.loss), 
        oracle=oracle,
        optimizer=optimizer,
        learning_policy=learning_policy
    )

    return cfg


def preparation(coach):
    from src.utils import TrackMeter, ImageMeter, getLogger
    from src.dict2obj import Config
    logger = getLogger()
    loss_logger = Config(
        train=TrackMeter("Train"),
        valid=TrackMeter("Valid")
    )

    loss_logger.plotter = ImageMeter(*loss_logger.values(), title="Loss")

    @timemeter("Evaluation")
    def evaluate(dataloader, prefix='Valid', epoch=8888):
        loss = coach.evaluate(dataloader)
        logger.info(f"{prefix} >>> [Loss: {loss:.6f}]")
        getattr(loss_logger, prefix.lower())(data=loss, T=epoch)
        return loss
    return loss_logger, evaluate


@timemeter("visual")
def visual(
    model, validloader, 
    info_path, log_path,
    device=DEVICE, epoch=8888
):
    import os
    from freeplot.base import FreePlot
    from freeplot.utils import export_pickle
    from src.utils import AverageMeter
    from src.loss_zoo import mse_loss
    loss_meter = AverageMeter("Loss")
    loss_func = mse_loss
    
    model.eval()
    running_target = {
        'x':[], 'y':[], 'g1':[], 'g2':[]
    }
    running_pred = {
        'x':[], 'y':[], 'g1':[], 'g2':[]
    }
    keys = ('x', 'y', 'g1', 'g2')
    for x, y, g1, g2 in validloader:
        x = x.to(device)
        y = y.to(device)
        g1 = g1.to(device)
        g2 = g2.to(device)

        target = oracle(x, g1, g2)
        x.requires_grad_(True)
        y_pred = model(x)
        g1_pred = torch.autograd.grad(
            y_pred, x, 
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True
        )[0]
        g2_pred = torch.autograd.grad(
            g1_pred, x, 
            grad_outputs=torch.ones_like(g1_pred),
            retain_graph=False
        )[0]
        x.requires_grad_(False)
        pred = oracle(x, g1_pred, g2_pred)
        loss = loss_func(pred, target)

        loss_meter.update(loss.item(), x.size(0), mode="mean")
        for key, value in zip(keys, (x, y, g1, g2)):
            running_target[key].append(value.clone().detach().cpu())
        for key, value in zip(keys, (x, y_pred, g1_pred, g2_pred)):
            running_pred[key].append(value.clone().detach().cpu())
    for key, value in running_target.items():
        running_target[key] = torch.cat(value, dim=0).numpy()
    for key, value in running_pred.items():
        running_pred[key] = torch.cat(value, dim=0).numpy()
    
    titles = ('y', 'g1', 'g2', 'y+', 'g1+', 'g2+')
    fp = FreePlot((2, 3), (7, 4), titles=titles, sharey=False, dpi=200, latex=False)
    for title in titles[:3]:
        x = running_target['x']
        y1 = running_target[title]
        y2 = running_pred[title]
        fp.lineplot(x, y1, index=title, label='Target')
        fp.lineplot(x, y2, index=title, label='Pred')
    for title in titles[:3]:
        x = running_target['x']
        y1 = running_target[title]
        y2 = running_pred[title]
        fp.lineplot(x, y1, index=title+'+', label='Target')
        fp.lineplot(x, y2, index=title+'+', label='Pred')
        fp.set_lim([-3, 3], index=title+'+', axis='y')
    data = {
        'real': running_target,
        'pred': running_pred
    }
    fp.set_title()
    fp[0, 0].legend()
    fp.savefig(os.path.join(log_path, f"visual_{epoch}.png"))
    return data



@timemeter("Main")
def main(
    coach,
    trainloader, validloader, boundary,
    start_epoch, 
    info_path, log_path
):  

    import os
    from src.utils import save_checkpoint
    from freeplot.utils import export_pickle

    for epoch in range(start_epoch, opts.epochs):
        
        #### Update the threshold ####
        if epoch == 0:
            _, beta = coach.updata_beta(validloader, opts)
        elif (epoch + 1) % 10 == 0:
            beta, _ = coach.updata_beta(validloader, opts)
            # beta = max(beta, 5)
            
        if epoch % SAVE_FREQ == 0:
            save_checkpoint(info_path, coach.model, coach.optimizer, coach.learning_policy, epoch)

        if epoch % opts.eval_freq == 0:
            if opts.eval_valid:
                visual(coach.model, validloader, info_path, log_path, coach.device, epoch)

        running_loss = coach.train(trainloader, boundary, beta = beta, leverage=opts.leverage, epoch=epoch)

    # save the model
    coach.save(info_path)


if __name__ ==  "__main__":
    from src.utils import readme
    cfg = load_cfg()
    opts.log_path = cfg.log_path
    readme(cfg.info_path, opts)
    readme(cfg.log_path, opts, mode="a")

    main(**cfg)

