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
from rich.table import Table
from rich.box import SIMPLE

install()
console = Console()

from .citations import Citable
from .util import copy_func, call_func, change_typer_to_defaults, add_kwargs
from .params import Param
from .callbacks import FastAppWandbCallback, FastAppMlflowCallback

bibtex_dir = Path(__file__).parent / "bibtex"


class FastAppInitializationError(Exception):
    pass


class FastApp(Citable):
    fastapp_initialized = False
    extra_params = None
    fine_tune = False

    def __init__(self):
        super().__init__()

        # Make deep copies of methods so that we can change the function signatures dynamically
        self.fit = self.copy_method(self.fit)
        self.train = self.copy_method(self.train)
        self.dataloaders = self.copy_method(self.dataloaders)
        self.model = self.copy_method(self.model)
        self.pretrained_location = self.copy_method(self.pretrained_location)
        self.show_batch = self.copy_method(self.show_batch)
        self.tune = self.copy_method(self.tune)
        self.pretrained_local_path = self.copy_method(self.pretrained_local_path)
        self.learner_kwargs = self.copy_method(self.learner_kwargs)
        self.learner = self.copy_method(self.learner)
        self.__call__ = self.copy_method(self.__call__)
        self.validate = self.copy_method(self.validate)
        self.callbacks = self.copy_method(self.callbacks)
        self.prepare_inference = self.copy_method(self.prepare_inference)

        # Add keyword arguments to the signatures of the methods used in the CLI
        add_kwargs(to_func=self.learner, from_funcs=[self.learner_kwargs, self.dataloaders, self.model])
        add_kwargs(to_func=self.train, from_funcs=[self.learner, self.fit, self.callbacks])
        add_kwargs(to_func=self.show_batch, from_funcs=self.dataloaders)
        add_kwargs(to_func=self.tune, from_funcs=self.train)
        add_kwargs(to_func=self.pretrained_local_path, from_funcs=self.pretrained_location)
        add_kwargs(to_func=self.prepare_inference, from_funcs=[self.pretrained_local_path, self.inference_dataloader])
        add_kwargs(to_func=self.__call__, from_funcs=self.prepare_inference)
        add_kwargs(to_func=self.validate, from_funcs=self.prepare_inference)

        # Make copies of methods to use just for the CLI
        self.train_cli = self.copy_method(self.train)
        self.show_batch_cli = self.copy_method(self.show_batch)
        self.tune_cli = self.copy_method(self.tune)
        self.pretrained_local_path_cli = self.copy_method(self.pretrained_local_path)
        self.infer_cli = self.copy_method(self.__call__)
        self.validate_cli = self.copy_method(self.validate)

        # Remove params from defaults in methods not used for the cli
        change_typer_to_defaults(self.fit)
        change_typer_to_defaults(self.model)
        change_typer_to_defaults(self.learner_kwargs)
        change_typer_to_defaults(self.learner)
        change_typer_to_defaults(self.callbacks)
        change_typer_to_defaults(self.train)
        change_typer_to_defaults(self.show_batch)
        change_typer_to_defaults(self.tune)
        change_typer_to_defaults(self.pretrained_local_path)
        change_typer_to_defaults(self.__call__)
        change_typer_to_defaults(self.validate)
        change_typer_to_defaults(self.dataloaders)
        change_typer_to_defaults(self.pretrained_location)

        # Store a bool to let the app know later on (in self.assert_initialized)
        # that __init__ has been called on this parent class
        self.fastapp_initialized = True

    def __str__(self):
        return self.__class__.__name__

    def get_bibtex_files(self):
        return [bibtex_dir / "fastai.bib", bibtex_dir / "fastapp.bib"]

    def copy_method(self, method):
        return MethodType(copy_func(method.__func__), self)

    def pretrained_location(self) -> Union[str, Path]:
        """
        The location of a pretrained model.

        It can be a URL, in which case it will need to be downloaded.
        Or it can be part of the package bundle in which case,
        it needs to be a relative path from directory which contains the code which defines the app.

        This function by default returns an empty string.
        Inherited classes need to override this method to use pretrained models.

        Returns:
            Union[str, Path]: The location of the pretrained model.
        """
        return ""

    def pretrained_local_path(
        self,
        pretrained: str = Param(default=None, help="The location (URL or filepath) of a pretrained model."),
        reload: bool = Param(
            default=False,
            help="Should the pretrained model be downloaded again if it is online and already present locally.",
        ),
        **kwargs,
    ) -> Path:
        """
        The local path of the pretrained model.

        If it is a URL, then it is downloaded.
        If it is a relative path, then this method returns the absolute path to it.

        Args:
            pretrained (str, optional): The location (URL or filepath) of a pretrained model. If it is a relative path, then it is relative to the current working directory. Defaults to using the result of the `pretrained_location` method.
            reload (bool, optional): Should the pretrained model be downloaded again if it is online and already present locally. Defaults to False.

        Raises:
            FileNotFoundError: If the file cannot be located in the local environment.

        Returns:
            Path: The absolute path to the model on the local filesystem.
        """
        if pretrained:
            location = pretrained
            base_dir = Path.cwd()
        else:
            location = str(call_func(self.pretrained_location, **kwargs))
            module = inspect.getmodule(self)
            base_dir = Path(module.__file__).parent.resolve()

        if not location:
            raise FileNotFoundError(f"Please pass in a pretrained model.")

        # Check if needs to be downloaded
        if location.startswith("http"):
            # TODO get user cache dir
            cached_download(location, user_cache_dir, reload)
        else:
            path = Path(location)
            if not path.is_absolute():
                path = base_dir / path

        if not path or not path.is_file():
            raise FileNotFoundError(f"Cannot find pretrained model at '{path}'")

        return path

    def prepare_source(self, data):
        return data

    def output_results(self, results, **kwargs):
        print(results)

    def inference_dataloader(self, learner, **kwargs):
        dataloader = learner.dls.test_dl(**kwargs)
        return dataloader

    def prepare_inference(self, **kwargs):
        # Open the exported learner from a pickle file
        path = call_func(self.pretrained_local_path, **kwargs)
        learner = load_learner(path)

        # Create a dataloader for inference
        dataloader = call_func(self.inference_dataloader, learner, **kwargs)

        return learner, dataloader

    def validate(self, **kwargs):
        learner, dataloader = call_func(self.prepare_inference, **kwargs)

        table = Table(title="Validation", box=SIMPLE)

        values = learner.validate(dl=dataloader)
        names = [learner.recorder.loss.name] + [metric.name for metric in learner.metrics]
        result = {name: value for name, value in zip(names, values)}

        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for name, value in result.items():
            table.add_row(name, str(value))

        console.print(table)

        return result

    def __call__(self, **kwargs):

        learner, dataloader = call_func(self.prepare_inference, **kwargs)
        results = learner.get_preds(dl=dataloader, reorder=False, with_decoded=False, act=self.activation())

        # Output results
        call_func(self.output_results, results, **kwargs)

        return results

    @classmethod
    def main(cls):
        """
        Creates an instance of this class and runs the command-line interface.
        """
        cli = cls.click()
        return cli()

    @classmethod
    def click(cls):
        """
        Creates an instance of this class and returns the click object for the command-line interface.
        """
        self = cls()
        cli = self.cli()
        return cli

    def assert_initialized(self):
        """
        Asserts that this app has been initialized.

        All sub-classes of FastApp need to call super().__init__() if overriding the __init__() function.

        Raises:
            FastAppInitializationError: If the app has not been properly initialized.
        """
        if not self.fastapp_initialized:
            raise FastAppInitializationError(
                """The initialization function for this FastApp has not been called.
                Please ensure sub-classes of FastApp call 'super().__init__()'"""
            )

    def version(self, verbose: bool = False):
        """
        Prints the version of the package that defines this app.

        Used in the command-line interface.

        Args:
            verbose (bool, optional): Whether or not to print to stdout. Defaults to False.

        Raises:
            Exception: If it cannot find the package.

        """
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

        params, _, _ = get_params_convertors_ctx_param_name_from_function(self.validate_cli)
        command = click.Command(
            name="validate",
            callback=self.validate_cli,
            params=params,
        )
        typer_click_object.add_command(command, "validate")

        params, _, _ = get_params_convertors_ctx_param_name_from_function(self.infer_cli)
        command = click.Command(
            name="infer",
            callback=self.infer_cli,
            params=params,
        )
        typer_click_object.add_command(command, "infer")

        command = click.Command(
            name="bibliography",
            callback=self.print_bibliography,
        )
        typer_click_object.add_command(command, "bibliography")

        command = click.Command(
            name="bibtex",
            callback=self.print_bibtex,
        )
        typer_click_object.add_command(command, "bibtex")

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
        fp16: bool = Param(
            default=True,
            help="Whether or not the floating-point precision of learner should be set to 16 bit.",
        ),
        **kwargs,
    ) -> Learner:
        """
        Creates a fastai learner object.
        """
        console.print("Building dataloaders", style="bold")
        dataloaders = call_func(self.dataloaders, **kwargs)

        # Allow the dataloaders to go to GPU so long as it hasn't explicitly been set as a different device
        if dataloaders.device is None:
            dataloaders.cuda()  # This will revert to CPU if cuda is not available

        console.print("Building model", style="bold")
        model = call_func(self.model, **kwargs)

        console.print("Building learner", style="bold")
        learner_kwargs = call_func(self.learner_kwargs, **kwargs)
        build_learner_func = self.build_learner_func()
        learner = build_learner_func(
            dataloaders,
            model,
            **learner_kwargs,
        )

        learner.training_kwargs = kwargs

        if fp16:
            console.print("Setting floating-point precision of learner to 16 bit", style="red")
            learner = learner.to_fp16()

        return learner

    def learner_kwargs(
        self,
        output_dir: Path = Param("./outputs", help="The location of the output directory."),
        **kwargs,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        # callbacks = call_func(self.callbacks, **kwargs)

        return dict(
            loss_func=self.loss_func(),
            metrics=self.metrics(),
            path=output_dir,
            # cbs=callbacks,
        )

    def loss_func(self):
        """The loss function. If None, then fastai will use the default loss function if it exists for this model."""
        return None

    def activation(self):
        """The activation for the last layer. If None, then fastai will use the default activiation of the loss if it exists."""
        return None

    def metrics(self) -> List:
        """
        The list of metrics to use with this app.

        By default this list is empty. This method should be subclassed to add metrics in child classes of FastApp.

        Returns:
            List: The list of metrics.
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
        project_name: str = Param(default=None, help="The name for this project for logging purposes."),
        run_name: str = Param(default=None, help="The name for this particular run for logging purposes."),
        notes: str = Param(None, help="A longer description of the run for logging purposes."),
        tag: List[str] = Param(
            None, help="A tag for logging purposes. Multiple tags can be added each introduced with --tag."
        ),
        wandb: bool = Param(default=False, help="Whether or not to use 'Weights and Biases' for logging."),
        wandb_mode: str = Param(default="online", help="The mode for 'Weights and Biases'."),
        wandb_dir: Path = Param(None, help="The location for 'Weights and Biases' output."),
        wandb_entity: str = Param(None, help="An entity is a username or team name where you're sending runs."),
        wandb_group: str = Param(None, help="Specify a group to organize individual runs into a larger experiment."),
        wandb_job_type: str = Param(
            None,
            help="Specify the type of run, which is useful when you're grouping runs together into larger experiments using group.",
        ),
        mlflow: bool = Param(default=False, help="Whether or not to use MLflow for logging."),
    ) -> List:
        """
        The list of callbacks to use with this app in the fastai training loop.

        Args:
            project_name (str): The name for this project for logging purposes. If no name is given then the name of the app is used.

        Returns:
            list: The list of callbacks.
        """
        callbacks = [CSVLogger()]
        monitor = self.monitor()
        if monitor:
            callbacks.append(SaveModelCallback(monitor=monitor))

        if wandb:
            callback = FastAppWandbCallback(
                app=self,
                project_name=project_name,
                name=run_name,
                mode=wandb_mode,
                dir=wandb_dir,
                entity=wandb_entity,
                group=wandb_group,
                job_type=wandb_job_type,
                notes=notes,
                tags=tag,
            )
            callbacks.append(callback)
            self.add_bibtex_file(bibtex_dir / "wandb.bib")  # this should be in the callback

        if mlflow:
            callbacks.append(FastAppMlflowCallback(app=self, experiment_name=project_name))
            self.add_bibtex_file(bibtex_dir / "mlflow.bib")  # this should be in the callback

        return callbacks

    def show_batch(self, **kwargs):
        dataloaders = call_func(self.dataloaders, **kwargs)
        dataloaders.show_batch()

    def train(
        self,
        distributed: bool = Param(default=False, help="If the learner is distributed."),
        **kwargs,
    ) -> Learner:
        """
        Trains a model for this app.

        Args:
            epochs (int, optional): The number of epochs. Defaults to 20.
            lr_max (float, optional): The max learning rate. Defaults to 1e-4.
            distributed (bool, optional): If the learner is distributed. Defaults to Param(default=False, help="If the learner is distributed.").

        Returns:
            Learner: The fastai Learner object created for training.
        """
        callbacks = call_func(self.callbacks, **kwargs)
        learner = call_func(self.learner, **kwargs)

        self.print_bibliography(verbose=True)

        with learner.distrib_ctx() if distributed == True else nullcontext():
            call_func(self.fit, learner, callbacks, **kwargs)

        learner.export()

        return learner

    def fit(
        self,
        learner,
        callbacks,
        epochs: int = Param(default=20, help="The number of epochs."),
        freeze_epochs: int = Param(
            default=3,
            help="The number of epochs to train when the learner is frozen and the last layer is trained by itself. Only if `fine_tune` is set on the app.",
        ),
        learning_rate: float = Param(
            default=1e-4, help="The base learning rate (when fine tuning) or the max learning rate otherwise."
        ),
        **kwargs,
    ):
        if self.fine_tune:
            return learner.fine_tune(
                epochs, freeze_epochs=freeze_epochs, base_lr=learning_rate, cbs=callbacks, **kwargs
            )  # hack

        return learner.fit_one_cycle(epochs, lr_max=learning_rate, cbs=callbacks, **kwargs)

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
