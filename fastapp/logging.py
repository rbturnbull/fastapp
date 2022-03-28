from pathlib import Path
import wandb
from fastcore.meta import delegates
from fastai.callback.wandb import WandbCallback
from fastai.learner import Learner

from .params import Param
from .callbacks import WandbCallbackTime


class WandbMixin(object):
    def __init__(self):
        super().__init__()
        delegates(to=self.init_run)(self.train)

    def init_run(
        self, 
        output_dir, 
        run_name,
        upload_model = Param(default=False, help="If true, logs model to WandB project"),
        **kwargs
    ):
        self.upload_model = upload_model
        
        if not run_name:
            run_name = Path(output_dir).name
        self.run = wandb.init(
            project=self.project_name(), 
            name=run_name,
            reinit=True,
            config=dict(
                **kwargs,
            )
        )

    def log(self, param):
        wandb.log(param)
    
    def log_artifact(self,artifact_path, artifact_name, artifact_type, upload = False, **kwargs):
        model_artifact = wandb.Artifact(artifact_name, type=artifact_type, **kwargs)
        if upload == True:
            model_artifact.add_file(artifact_path)
        else:
            model_artifact.add_reference(artifact_path)
        self.run.log_artifact(model_artifact)

    def logging_callbacks(self, callbacks):
        wandb_callback = WandbCallback(log_preds=False)
        callbacks.extend([wandb_callback, WandbCallbackTime(wandb_callback=wandb_callback)])
        return callbacks

    def save_model(self, learner: Learner, run_name):
        super().save_model(learner, run_name)

        model_path = learner.path/learner.model_dir/run_name
        # import pdb;pdb.set_trace()
        self.log_artifact(model_path, run_name, 'model',upload=self.upload_model)
        
    def tune(
        self,
        id: str=None,
        name: str=None,
        method: str="random", # Should be enum
        runs: int=1,
        min_iter: int=None,
        **kwargs,
    ):
        if not name:
            name = f"{self.project_name()}-tuning"
        self.init_run(run_name=name)
        if not id:

            parameters_config = dict()
            tuning_params = self.tuning_params()
            for key, value in tuning_params.items():
                if key in kwargs and kwargs[key] is None:
                    parameters_config[key] = value.config()
            
            sweep_config = {
                "name" : name,
                "method" : method,
                "parameters" : parameters_config,
            }
            if self.monitor():
                sweep_config['metric'] = dict(name=self.monitor(), goal=self.goal())

            if min_iter:
                sweep_config['early_terminate'] = dict(type="hyperband", min_iter=min_iter)

            console.print("Configuration for hyper-parameter tuning:", style="bold red")
            pprint(sweep_config)

            id = wandb.sweep(sweep_config, project=name)
            console.print(f"The wandb sweep id is: {id}", style="bold red")

        def agent_train():
            with wandb.init() as run:
                run_kwargs = dict(kwargs)
                run_kwargs.update(wandb.config)
                if 'output_dir' in run_kwargs:
                    run_kwargs['output_dir'] = Path(run_kwargs['output_dir'])/run.name

                console.print("Training with parameters:", style="bold red")
                pprint(run_kwargs)

                run_callback(self.train, run_kwargs)

        wandb.agent(id, function=agent_train, count=runs, project=name)

        return id