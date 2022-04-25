import sys
from contextlib import nullcontext
from pathlib import Path
from types import MethodType
import inspect
from typing import List, Optional, Union, Dict
from torch import nn
from fastai.learner import Learner, load_learner
from fastai.data.core import DataLoaders
from fastai.callback.schedule import fit_one_cycle
from fastai.distributed import distrib_ctx
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.progress import CSVLogger
import click
import typer
from typer.main import get_params_convertors_ctx_param_name_from_function
from rich.pretty import pprint
from rich.console import Console
from rich.traceback import install

install()
console = Console()

from .util import copy_func, run_callback, change_typer_to_defaults, add_kwargs
from .params import Param
from .callbacks import FastAppWandbCallback, FastAppMlflowCallback


class FastAppInitializationError(Exception):
    pass


class FastApp:
    fastapp_initialized = False
    extra_params = None

    def __init__(self):
        super().__init__()

        # Make deep copies of methods so that we can change the function signatures dynamically
        self.train = self.copy_method(self.train)
        self.dataloaders = self.copy_method(self.dataloaders)
        self.model = self.copy_method(self.model)
        self.pretrained_location = self.copy_method(self.pretrained_location)
        self.show_batch = self.copy_method(self.show_batch)
        self.tune = self.copy_method(self.tune)
        self.pretrained_local_path = self.copy_method(self.pretrained_local_path)
        self.__call__ = self.copy_method(self.__call__)
        self.callbacks = self.copy_method(self.callbacks)

        # Add keyword arguments to the signatures of the methods used in the CLI
        add_kwargs(to_func=self.train, from_funcs=[self.dataloaders, self.model, self.callbacks])
        add_kwargs(to_func=self.show_batch, from_funcs=self.dataloaders)
        add_kwargs(to_func=self.tune, from_funcs=self.train)
        add_kwargs(to_func=self.pretrained_local_path, from_funcs=self.pretrained_location)
        add_kwargs(to_func=self.__call__, from_funcs=self.pretrained_local_path)

        # Make copies of methods to use just for the CLI
        self.train_cli = self.copy_method(self.train)
        self.show_batch_cli = self.copy_method(self.show_batch)
        self.tune_cli = self.copy_method(self.tune)
        self.pretrained_local_path_cli = self.copy_method(self.pretrained_local_path)
        self.call_cli = self.copy_method(self.__call__)

        # Remove params from defaults in methods not used for the cli
        change_typer_to_defaults(self.model)
        change_typer_to_defaults(self.callbacks)
        change_typer_to_defaults(self.train)
        change_typer_to_defaults(self.show_batch)
        change_typer_to_defaults(self.tune)
        change_typer_to_defaults(self.pretrained_local_path)
        change_typer_to_defaults(self.__call__)
        change_typer_to_defaults(self.dataloaders)
        change_typer_to_defaults(self.pretrained_location)

        # Store a bool to let the app know later on (in self.assert_initialized)
        # that __init__ has been called on this parent class
        self.fastapp_initialized = True

    def __str__(self):
        return self.__class__.__name__

    def copy_method(self, method):
        return MethodType(copy_func(method.__func__), self)

    def pretrained_location(self) -> Union[str, Path]:
        return ""

    def pretrained_local_path(
        self,
        pretrained: str = Param(default=None, help="The location (URL or filepath) of a pretrained model."),
        reload: bool = Param(
            default=False,
            help="Should the pretrained model be downloaded again if it is online and already present locally.",
        ),
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
            with open(output, "w") as f:
                f.write(results)

    def test_dataloader(self, learner, prepared_data):
        dataloader = learner.dls.test_dl(prepared_data)
        return dataloader

    def __call__(self, data, output: str = "", **kwargs):
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
        cli = cls.click()
        return cli()

    @classmethod
    def click(cls):
        self = cls()
        cli = self.cli()
        return cli

    def assert_initialized(self):
        if not self.fastapp_initialized:
            raise FastAppInitializationError(
                """The initialization function for this FastApp has not been called.
                Please ensure sub-classes of FastApp call 'super().__init__()'"""
            )

    def version(self, verbose: bool = False):
        if verbose:
            from importlib import metadata

            module = inspect.getmodule(self)
            package = ""
            if module.__package__:
                package = module.__package__.split('.')[0]
            else:
                path = Path(module.__file__).parent
                while path.name:
                    try:
                        if metadata.distribution(path.name):
                            package = path.name
                            break
                    except Exception:
                        pass
                    path = path.parent

            if package:
                version = metadata.version(package)
                print(version)
            else:
                raise Exception("Cannot find package.")

            raise typer.Exit()

    def cli(self):
        """
        Returns a 'Click' object which defines the command-line interface of the app.
        """
        self.assert_initialized()

        cli = typer.Typer()

        @cli.callback()
        def base_callback(
            version: Optional[bool] = typer.Option(
                None,
                "--version",
                "-v",
                callback=self.version,
                is_eager=True,
                help="Prints the current version.",
            ),
        ):
            pass

        typer_click_object = typer.main.get_command(cli)

        train_params, _, _ = get_params_convertors_ctx_param_name_from_function(self.train_cli)
        train_command = click.Command(
            name="train",
            callback=self.train_cli,
            params=train_params,
        )
        typer_click_object.add_command(train_command, "train")

        show_batch_params, _, _ = get_params_convertors_ctx_param_name_from_function(self.show_batch_cli)
        command = click.Command(
            name="show-batch",
            callback=self.show_batch_cli,
            params=show_batch_params,
        )
        typer_click_object.add_command(command, "show-batch")

        params, _, _ = get_params_convertors_ctx_param_name_from_function(self.tune_cli)
        tuning_params = self.tuning_params()
        for param in params:
            if param.name in tuning_params:
                param.default = None
        command = click.Command(
            name="tune",
            callback=self.tune_cli,
            params=params,
        )
        typer_click_object.add_command(command, "tune")

        params, _, _ = get_params_convertors_ctx_param_name_from_function(self.call_cli)
        command = click.Command(
            name="predict",
            callback=self.call_cli,
            params=params,
        )
        typer_click_object.add_command(command, "predict")

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
        raise NotImplementedError(
            f"Please ensure that the 'dataloaders' method is implemented in {self.__class__.__name__}."
        )

    def model(self) -> nn.Module:
        raise NotImplementedError(f"Please ensure that the 'model' method is implemented in {self.__class__.__name__}.")

    def build_learner_func(self):
        return Learner

    def learner(
        self,
        dataloaders,
        output_dir: Path,
        fp16: bool = Param(
            default=True,
            help="Whether or not the floating-point precision of learner should be set to 16 bit.",
        ),
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
            learner = build_learner_func(
                dataloaders,
                model,
                loss_func=self.loss_func(),
                metrics=self.metrics(),
                path=output_dir,
            )

        else:
            learner = build_learner_func(
                dataloaders,
                model,
                loss_func=self.loss_func(),
                metrics=self.metrics(),
                path=output_dir,
            )

        if fp16:
            console.print("Setting floating-point precision of learner to 16 bit", style="red")
            learner = learner.to_fp16()

        return learner

    def loss_func(self):
        """The loss function. If None, then fastai will use the default loss function if it exists for this model."""
        return None

    def activation(self):
        """The activation for the last layer. If None, then fastai will use the default activiation of the loss if it exists."""
        return None

    def metrics(self) -> list:
        """
        The list of metrics to use with this app.

        By default this list is empty. This method should be subclassed to add metrics in child classes of FastApp.

        Returns:
            list: The list of metrics
        """
        return []

    def monitor(self) -> str:
        """
        The metric to optimize for when performing hyperparameter tuning.

        By default it returns 'valid_loss'.
        """
        return "valid_loss"

    def goal(self) -> str:
        """
        Sets the optimality direction when evaluating the metric from `monitor`.

        By default it produces the same behaviour as fastai callbacks (fastai.callback.tracker)
        ie. it is set to "minimize" if the monitor metric has the string 'loss' or 'err' otherwise it is "maximize".

        If the monitor is empty then this function returns None.
        """
        monitor = self.monitor()
        if not monitor or not isinstance(monitor, str):
            return None

        return "minimize" if ("loss" in monitor) or ("err" in monitor) else "maximize"

    def callbacks(
        self,
        wandb: bool = Param(default=False, help="Whether or not to use 'Weights and Biases' for logging."),
        wandb_mode: str = Param(default="online", help="The mode for 'Weights and Biases'."),
        wandb_dir: Path = Param(None, help="The location for 'Weights and Biases' output."),
        mlflow: bool = Param(default=False, help="Whether or not to use MLflow for logging."),
    ) -> List:
        """
        The list of callbacks to use with this app in the fastai training loop.

        Returns:
            list: The list of callbacks.
        """
        callbacks = [CSVLogger()]
        monitor = self.monitor()
        if monitor:
            callbacks.append(SaveModelCallback(monitor=monitor))

        if wandb:
            callbacks.append(FastAppWandbCallback(app=self, mode=wandb_mode, dir=wandb_dir))

        if mlflow:
            callbacks.append(FastAppMlflowCallback(app=self))

        return callbacks

    def show_batch(self, **kwargs):
        dataloaders = run_callback(self.dataloaders, kwargs)
        dataloaders.show_batch()

    def train(
        self,
        output_dir: Path = Path("./outputs"),
        epochs: int = Param(default=20, help="The number of epochs."),
        lr_max: float = Param(default=1e-4, help="The max learning rate."),
        distributed: bool = Param(default=False, help="If the learner is distributed."),
        **kwargs,
    ) -> Learner:
        """
        Trains a model for this app.

        Args:
            output_dir (Path, optional): _description_. Defaults to Path("./outputs").
            epochs (int, optional): The number of epochs. Defaults to 20.
            lr_max (float, optional): The max learning rate. Defaults to 1e-4.
            distributed (bool, optional): _description_. Defaults to Param(default=False, help="If the learner is distributed.").
            run_name (str): The name for this run for logging purposes. If no name is given then the name of the output directory is used.

        Returns:
            Learner: The fastai Learner object created for training.
        """
        dataloaders = run_callback(self.dataloaders, kwargs)

        # Allow the dataloaders to go to GPU so long as it hasn't explicitly been set as a different device
        if dataloaders.device is None:
            dataloaders.cuda()  # This will revert to CPU if cuda is not available

        learner = self.learner(dataloaders, output_dir=output_dir, **kwargs)

        callbacks = run_callback(self.callbacks, kwargs)

        with learner.distrib_ctx() if distributed == True else nullcontext():
            learner.fit_one_cycle(epochs, lr_max=lr_max, cbs=callbacks)
            # more flexibility needs to be added here for other types of training loops

        return learner

    def project_name(self) -> str:
        """
        The name to use for a project for logging purposes.

        The default is to use the class name.
        """
        return self.__class__.__name__

    def tune(
        self,
        runs: int = Param(default=1, help="The number of runs to attempt to train the model."),
        engine: str = Param(
            default="wandb", help="The optimizer to use to perform the hyperparameter tuning."
        ),  # should be enum
        id: str = Param(default="", help="The ID of this hyperparameter tuning job if being used by multiple agents."),
        name: str = Param(
            default="",
            help="An informative name for this hyperparameter tuning job. If empty, then it creates a name from the project name.",
        ),
        wandb_method: str = Param(
            default="random", help="The optimizer to use to perform the hyperparameter tuning."
        ),  # should be enum
        min_iter: int = Param(
            default=None,
            help="The minimum number of iterations if using early termination. If left empty, then early termination is not used.",
        ),
        **kwargs,
    ):
        if not name:
            name = f"{self.project_name()}-tuning"

        if engine == "wandb":
            from .tuning.wandb import wandb_tune

            return wandb_tune(
                self,
                runs=runs,
                sweep_id=id,
                name=name,
                method=wandb_method,
                min_iter=min_iter,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"Optimizer engine {engine} not implemented.")
