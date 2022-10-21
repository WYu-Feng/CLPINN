


from multiprocessing import reduction
from typing import Callable, Dict, Iterable
import torch
import torch.nn as nn
import os

from .utils import AverageMeter, ProgressMeter, timemeter, getLogger
from .loss_zoo import mse_loss
from .config import DEVICE, SAVED_FILENAME, PRE_BEST
import numpy as np

class Coach:
    
    def __init__(
        self, model: nn.Module,
        loss_func: Callable, 
        oracle: Callable,
        optimizer: torch.optim.Optimizer, 
        learning_policy: "learning rate policy",
        device: torch.device = DEVICE,
    ):
        self.model = model
        self.device = device
        self.loss_func = loss_func
        self.oracle = oracle
        self.optimizer = optimizer
        self.learning_policy = learning_policy
        self.loss = AverageMeter("Loss")
        self.progress = ProgressMeter(self.loss)

        self._best = float('inf')

   
    def save_best(self, mse: float, path: str, prefix: str = PRE_BEST):
        if mse < self._best:
            self._best = mse
            self.save(path, '_'.join((prefix, SAVED_FILENAME)))
            return 1
        else:
            return 0

    def check_best(
        self, mse: float,
        path: str, epoch: int = 8888
    ):
        logger = getLogger()
        if self.save_best(mse, path):
            logger.debug(f"[Coach] Saving the best nat ({mse:.6f}) model at epoch [{epoch}]")
        
    def save(self, path, epoch):
        torch.save(self.model.state_dict(), os.path.join(path, '{:}_'.format(epoch) + SAVED_FILENAME))

    def updata_beta(self, validloader, opts):
        loss_function = nn.MSELoss(reduction='none')
        for i, (x, y, g1, g2) in enumerate(validloader):
            x = x.to(self.device)
            y = y.to(self.device)
            g1 = g1.to(self.device)
            g2 = g2.to(self.device)
            
            target = self.oracle(x, y, g1, g2)
            x.requires_grad_(True)
            y_pred = self.model(x)
            g1_pred = torch.autograd.grad(
                y_pred, x, 
                grad_outputs=torch.ones_like(y_pred),
                create_graph=True
            )[0]
            g2_pred = torch.autograd.grad(
            g1_pred, x, 
            grad_outputs=torch.ones_like(g1_pred),
            create_graph=True
            )[0]
            pred = self.oracle(x, y_pred, g1_pred, g2_pred)
            loss = loss_function(pred, target)
            loss_g = torch.autograd.grad(
            loss, x, 
            grad_outputs=torch.ones_like(loss),
            create_graph=True
            )[0]
            x.requires_grad_(False)
        
        loss_g = loss_g.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        beta_loss_list = list()
        for loss_g_item, loss_item in zip(loss_g, loss):
            if abs(loss_g_item) < opts.G:
                beta_loss_list.append(loss_item)
        max_beta = np.max(beta_loss_list)
        mean_beta = np.mean(beta_loss_list)
        return max_beta, mean_beta
    

    def train(self, trainloader, boundary, beta, leverage, epoch = 8888):
        self.progress.step() # reset the meter
        self.model.train()
        bx, by, bg1, bg2 = boundary['x'], boundary['y'], boundary['g1'], boundary['g2']
        for x, y, g1, g2 in trainloader:
            x = x.to(self.device)
            beta_batch = torch.tensor([beta for _ in range(x.size(0))]).to(self.device)
            y = y.to(self.device)
            g1 = g1.to(self.device)
            g2 = g2.to(self.device)
            bx = bx.to(self.device)
            by = by.to(self.device)

            def closure():
                target = self.oracle(x, y, g1, g2)
                x.requires_grad_(True)
                y_pred = self.model(x)
                g1_pred = torch.autograd.grad(
                    y_pred, x, 
                    grad_outputs=torch.ones_like(y_pred),
                    create_graph=True
                )[0]
                g2_pred = torch.autograd.grad(
                g1_pred, x, 
                grad_outputs=torch.ones_like(g1_pred),
                create_graph=True
                )[0]
                x.requires_grad_(False)
                pred = self.oracle(x, y_pred, g1_pred, g2_pred)
                loss = self.loss_func(pred, target, reduction='none')
                
                #### Reweighting of loss 
                loss = torch.mean(torch.min(loss, beta_batch))
                
                by_pred = self.model(bx)
                bloss = mse_loss(by_pred, by, reduction="mean")
                loss = loss + bloss * leverage
                self.optimizer.zero_grad()
                loss.backward()
                return loss

            loss = self.optimizer.step(closure)
            self.loss.update(loss.item(), x.size(0), mode="mean")

        self.progress.display(epoch=epoch) 
        self.learning_policy.step() # update the learning rate
        return self.loss.avg
    
