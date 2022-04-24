"""The logging module contains prebuilt mixin classes
    for logging model runs to different mlOps metadata and artifact stores.

    At minimum, these classes need to contain the following methods:

    `init_run` - this initialises a run in the mlOps framework for the training run 
    to log too

    `log` - function for logging a parameter or metric. Overwrites a function that just prints in to stdout

    `log_artifact`- a function to log file artifacts (files and models etc) to the mlOps artifact store

    `save_model` - function that will save the model weights and log them to an artifact store, overwrites the learner.save function in the base app.

    Other optional methods that may be added updated:

    `logging_callbacks` - updates callback list to include mlOps specific callbacks

    `tune` - a rewritten tuning function if mlOps packages includes that



    Currently, a mixin for using Weights and Biases implemented, and includes
    a tuning function for using Weighths and Biases sweeps for hyperparameter tuning

"""


from pathlib import Path
import wandb
from fastcore.meta import delegates
from fastai.callback.wandb import WandbCallback
from fastai.learner import Learner
from typing import Union, Optional
from .params import Param
from .callbacks import WandbCallbackTime

from rich.pretty import pprint
from rich.console import Console
from rich.traceback import install

from .apps import run_callback

install()
console = Console()
