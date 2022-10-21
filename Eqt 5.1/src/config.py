







# Here are some basic settings.
# It could be overwritten if you want to specify
# some configs. However, please check the correspoding
# codes in loadopts.py.



import torch
import logging
from .dict2obj import Config



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = "../data" # the path saving the data
DOWNLOAD = False # whether to download the data

SAVED_FILENAME = "paras.pt" # the filename of saved model paramters
PRE_BEST = "best"
INFO_PATH = "./infos/{method}/{dataset}-{model}/{description}"
LOG_PATH = "./logs/{method}/{dataset}-{model}/{description}-{time}"
TIMEFMT = "%m%d%H"

# logger
LOGGER = Config(
    name='RFK', filename='log.txt', level=logging.DEBUG,
    filelevel=logging.DEBUG, consolelevel=logging.INFO,
    formatter=Config(
        filehandler=logging.Formatter('%(asctime)s:\t%(message)s'),
        consolehandler=logging.Formatter('%(message)s')
    )
)

# the seed for validloader preparation
VALIDSEED = 1

# default transforms
TRANSFORMS = None

# env settings
NUM_WORKERS = 3
PIN_MEMORY = True


# the settings of optimizers of which lr could be pointed
# additionally.
OPTIMS = {
    "sgd": Config(lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=False, prefix="SGD:"),
    "adam": Config(lr=0.01, betas=(0.9, 0.999), weight_decay=0., prefix="Adam:"),
    'lbfgs': Config(lr=1, max_iter=20, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100, prefix="LBFGS:")
}


# the learning schedule can be added here
LEARNING_POLICY = {
    "null": (
        "StepLR",
        Config(
            step_size=9999999999999,
            gamma=1,
            prefix="Null leaning policy will be applied:"
        )
    ),
   "Pang2021ICLR": (
        "MultiStepLR",
        Config(
            milestones=[100, 105],
            gamma=0.1,
            prefix="Pang2020ICLR leaning policy will be applied:"
        )
    ),
    "Rice2020ICML": (
        "MultiStepLR",
        Config(
            milestones=[100, 150],
            gamma=0.1,
            prefix="Rice2020ICML leaning policy will be applied:"
        )
    ),
    "STD": (
        "MultiStepLR",
        Config(
            milestones=[82, 123],
            gamma=0.1,
            prefix="STD leaning policy will be applied:"
        )
    ),
    "STD-wrn": (
        "MultiStepLR",
        Config(
            milestones=[60, 120, 160],
            gamma=0.2,
            prefix="STD-wrn leaning policy will be applied:"
        )
    ),
    "AT":(
        "MultiStepLR",
        Config(
            milestones=[102, 154],
            gamma=0.1,
            prefix="AT learning policy, an official config:"
        )
    ),
    "TRADES":(
        "MultiStepLR",
        Config(
            milestones=[75, 90, 100],
            gamma=0.1,
            prefix="TRADES learning policy, an official config:"
        )
    ),
    "TRADES-M":(
        "MultiStepLR",
        Config(
            milestones=[55, 75, 90],
            gamma=0.1,
            prefix="TRADES-M learning policy, an official config for MNIST:"
        )
    ),
    "cosine":(   
        "CosineAnnealingLR",   
        Config(          
            T_max=200,
            eta_min=0.,
            last_epoch=-1,
            prefix="cosine learning policy: T_max == epochs - 1:"
        )
    )
}






