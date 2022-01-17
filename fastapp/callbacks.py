import time
import wandb
from fastai.callback.core import Callback


class WandbCallbackTime(Callback):
    def __init__(self, wandb_callback):
        self.wandb_callback = wandb_callback

    def after_epoch(self):
        wandb.log({'time': time.time() - self.recorder.start_epoch}, step=self.wandb_callback._wandb_step)

