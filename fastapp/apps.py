from contextlib import nullcontext
from pathlib import Path
import inspect
from typing import List, Optional, Union, Dict

import wandb

from torch import nn
from fastcore.meta import delegates
from fastai.learner import Learner, load_learner
from fastai.vision.learner import cnn_learner
from fastai.data.core import DataLoaders
from fastai.callback.schedule import fit_one_cycle
from fastai.distributed import distrib_ctx
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.wandb import WandbCallback
from fastai.callback.progress import CSVLogger

import click
import typer
from typer.main import get_params_convertors_ctx_param_name_from_function
from typer.utils import get_params_from_function

from rich.traceback import install
from rich.pretty import pprint
from rich.console import Console

install()
console = Console()

from .params import Param
from .callbacks import WandbCallbackTime


def run_callback(callback, params):
    allowed_params, _, _ = get_params_convertors_ctx_param_name_from_function(callback)
    allowed_param_names = [p.name for p in allowed_params]
    kwargs = {key:value for key,value in params.items() if key in allowed_param_names}
    return callback(**kwargs)


def version_callback(value: bool):
    """
    Prints the current version.
    """
    if value:
        import importlib.metadata
        module_name = str(__name__).split(".")[0]
        version = importlib.metadata.version(module_name)
        console.print(version)
        raise typer.Exit()


class FastApp:
    extra_params = None

    def __init__(self):
        # All these would be better as decorators
        delegates(to=self.dataloaders, keep=True)(self.train)
        delegates(to=self.model)(self.train)

        delegates(to=self.dataloaders)(self.show_batch)

        delegates(to=self.train)(self.tune)

        delegates(to=self.pretrained_location)(self.pretrained_local_path)

        delegates(to=self.pretrained_local_path)(self.__call__)

    def pretrained_location(self) -> Union[str, Path]:
        return ""

    def pretrained_local_path(
        self, 
        pretrained:str = Param(default=None, help="The location (URL or filepath) of a pretrained model."), 
        reload:bool = Param(default=False, help="Should the pretrained model be downloaded again if it is online and already present locally."),
        **kwargs,
    ):
        if pretrained:
            location = pretrained
        else:
            location = str(run_callback(self.pretrained_location, kwargs))
        
        # Check if needs to be downloaded
        if location.startswith("http"):
            # TODO get user cache dir
            cached_download(location, user_cache_dir, reload)

        path = Path(location)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Cannot find pretrained model at '{path}'")

        return path

    def prepare_source(self, data):
        return data

    def output_results(self, results, data, prepared_data, output):
        print(results)
        if output:
            with open(output, 'w') as f:
                f.write(results)

    def test_dataloader(self, learner, prepared_data):
        dataloader = learner.dls.test_dl(prepared_data)
        return dataloader

    def __call__(self, data, output:str = "", **kwargs):
        path = run_callback(self.pretrained_local_path, kwargs)

        # open learner from pickled file
        learner = load_learner(path)

        # Classify results
        prepared_data = self.prepare_source(data)
        dataloader = self.test_dataloader(learner, prepared_data)
        results = learner.get_preds(dl=dataloader, reorder=False, with_decoded=False, act=self.activation())

        results = self.output_results(results, data, prepared_data, output)

        return results

    @classmethod
    def main(cls):
        self = cls()
        cli = self.cli()
        return cli()

    def cli(self):
        cli = typer.Typer()

        @cli.callback()
        def base_callback(
            version: Optional[bool] = typer.Option(None, "--version", "-v", callback=version_callback, is_eager=True, help="Prints the current version."), 
        ):
            pass

        typer_click_object = typer.main.get_command(cli)

        train_params, _, _ = get_params_convertors_ctx_param_name_from_function(self.train)
        train_command = click.Command(
            name='train',
            callback=self.train,
            params=train_params,
        )
        typer_click_object.add_command(train_command, 'train')

        show_batch_params, _, _ = get_params_convertors_ctx_param_name_from_function(self.show_batch)
        command = click.Command(
            name='show-batch',
            callback=self.show_batch,
            params=show_batch_params,
        )
        typer_click_object.add_command(command, 'show-batch')

        
        params, _, _ = get_params_convertors_ctx_param_name_from_function(self.tune)
        tuning_params = self.tuning_params()
        for param in params:
            if param.name in tuning_params:
                param.default = None
        command = click.Command(
            name='tune',
            callback=self.tune,
            params=params,
        )
        typer_click_object.add_command(command, 'tune')

        params, _, _ = get_params_convertors_ctx_param_name_from_function(self.__call__)
        command = click.Command(
            name='predict',
            callback=self.__call__,
            params=params,
        )
        typer_click_object.add_command(command, 'predict')

        return typer_click_object

    def tuning_params(self):
        tuning_params = {}
        signature = inspect.signature(self.tune)

        for key, value in signature.parameters.items():
            default_value = value.default
            if isinstance(default_value, Param) and default_value.tune == True:

                # Override annotation if given in typing hints
                if value.annotation:
                    default_value.annotation = value.annotation

                tuning_params[key] = default_value
        return tuning_params

    def dataloaders(self):
        raise NotImplementedError

    def model(self) -> nn.Module:
        raise NotImplementedError

    def build_learner_func(self):
        return Learner

    def learner(
        self,
        dataloaders,
        output_dir: Path,
        fp16:bool = Param(default=True, help="Whether or not the floating-point precision of learner should be set to 16 bit."),
        **params,
    ) -> Learner:
        """
        Creates a fastai learner object.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        console.print("Building model", style="bold")
        model = run_callback(self.model, params)

        console.print("Building learner", style="bold")
        build_learner_func = self.build_learner_func()
        if model is not None:
            learner = build_learner_func(dataloaders, model, loss_func=self.loss_func(), metrics=self.metrics(), path=output_dir)

        else:
            learner = build_learner_func(dataloaders, model, loss_func=self.loss_func(), metrics=self.metrics(), path=output_dir)


        if fp16:
            console.print("Setting floating-point precision of learner to 16 bit", style="red")
            learner = learner.to_fp16()

        return learner

    def loss_func(self):
        return None
    
    def activation(self):
        """ The activation for the last layer. If None, then fastai will use the default activiation of the loss if it exists. """
        return None

    def metrics(self):
        return []

    def monitor(self):
        return None

    def goal(self):
        """ 
        Sets the optimality direction when evaluating the metric from `monitor`.

        By default it is set to "minimize" if the monitor metric has the string 'loss' or 'err' otherwise it is "maximize".
        """
        monitor = self.monitor()
        # Compare fastai.callback.tracker
        return "minimize" if 'loss' in monitor or 'err' in monitor else "maximize"

    def callbacks(self):
        callbacks = [CSVLogger()]        
        monitor = self.monitor()
        if monitor:
            callbacks.append(SaveModelCallback(monitor=monitor))
        callbacks = self.logging_callbacks(callbacks)
        return callbacks

    def show_batch(self, **kwargs):
        dataloaders = run_callback(self.dataloaders, kwargs)
        dataloaders.show_batch()

    def train(
        self,
        output_dir: Path = Path("./outputs"),
        epochs:int = Param(default=20, help="The number of epochs."),
        lr_max:float = Param(default=1e-4, help="The max learning rate."),
        distributed:bool = Param(default=False, help="If the learner is distributed."),
        # wandb:bool = Param(default=False, help="If training should use Weights & Biases."),
        run_name:str = Param(default="", help="The name for this run in Weights & Biases. If no name is given then the name of the output directory is used."),
        **kwargs,
    ) -> Learner:
        
        kwargs_plus_lr  = kwargs.copy()
        kwargs_plus_lr['lr_max'] = lr_max
        
        self.init_run(run_name=run_name, output_dir=output_dir, **kwargs_plus_lr)

        dataloaders = run_callback(self.dataloaders, kwargs)

        learner = self.learner(dataloaders, output_dir=output_dir, **kwargs)

        with learner.distrib_ctx() if distributed == True else nullcontext():
            learner.fit_one_cycle(epochs, lr_max=lr_max, cbs=self.callbacks())
        
        self.save_model(learner, run_name)
        return learner

    def project_name(self):
        return self.__class__.__name__

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

    def init_run(self, run_name, output_dir, **kwargs):
        if not run_name:
            run_name = Path(output_dir).name
        print(f'from {self.project_name}')
        print(f'running {run_name}')
        print(f'with these parameters: \n {kwargs}')
    
    def log(self, param):
        print(param)
    
    def logging_callbacks(self, callbacks):
        return callbacks
    def save_model(self, learner, run_name):
        learner.save(run_name)
    


class WandbLoggingMixin(object):
    upload_model = Param(default=False, help="If true, logs model to WandB project")
    model_name = Param(default='trained_model', help='name of trained model artifact')

    def init_run(self, run_name, output_dir, **kwargs):

        
        
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
        super().save(learner, run_name)

        model_path = learner.path/learner.model_dir/run_name

        self.log_artifact(model_path, self.model_name, 'model',upload=self.upload_model)
        

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

class VisionApp(FastApp):

    def default_model_name(self):
        return "resnet18"

    def model(
        self,
        model_name:str = Param(default="", help="The name of a model architecture in torchvision.models (https://pytorch.org/vision/stable/models.html)."),
        pretrained:bool = Param(default=True, help="Whether or not to use the pretrained weights.")
    ):
        import torchvision.models as models

        if not model_name:
            model_name = self.default_model_name()

        if not hasattr(models, model_name):
            raise ValueError(f"Model '{model_name}' not recognized.")
        
        return getattr( models, model_name )(pretrained=pretrained)

    def build_learner_func(self):
        return cnn_learner