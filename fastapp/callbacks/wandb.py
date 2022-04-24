import time
import wandb
from fastai.callback.wandb import WandbCallback


class FastAppWandbCallback(WandbCallback):
    def __init__(self, app, output_dir, project_name=None, **kwargs):
        super().__init__()
        self.app = app
        if project_name is None:
            project_name = app.project_name()

        self.run = wandb.init(dir=output_dir, project=project_name, reinit=True, **kwargs)

    def after_epoch(self):
        super().after_epoch()

        # Record the length of time of each epoch
        wandb.log(
            {"time": time.time() - self.recorder.start_epoch},
            step=self._wandb_step,
        )
