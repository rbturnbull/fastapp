from contextlib import nullcontext
from pathlib import Path
import inspect
from typing import List, Optional, Union, Dict
from torch import nn
from fastcore.meta import delegates
from fastai.learner import Learner, load_learner
from fastai.data.core import DataLoaders
from fastai.callback.schedule import fit_one_cycle
from fastai.distributed import distrib_ctx
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.progress import CSVLogger
import click
import typer
from typer.main import get_params_convertors_ctx_param_name_from_function
from typer.utils import get_params_from_function
from rich.pretty import pprint
from rich.console import Console
from rich.traceback import install

install()
console = Console()

from .params import Param


def run_callback(callback, params):
    allowed_params, _, _ = get_params_convertors_ctx_param_name_from_function(callback)
    allowed_param_names = [p.name for p in allowed_params]
    kwargs = {key: value for key, value in params.items() if key in allowed_param_names}
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
        super().__init__()

        # All these would be better as decorator
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
        pretrained: str = Param(
            default=None, help="The location (URL or filepath) of a pretrained model."
        ),
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
        results = learner.get_preds(
            dl=dataloader, reorder=False, with_decoded=False, act=self.activation()
        )

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

    def cli(self):
        """
        Returns a 'Click' object which defines the command-line interface of the app.
        """
        cli = typer.Typer()

        @cli.callback()
        def base_callback(
            version: Optional[bool] = typer.Option(
                None,
                "--version",
                "-v",
                callback=version_callback,
                is_eager=True,
                help="Prints the current version.",
            ),
        ):
            pass

        typer_click_object = typer.main.get_command(cli)

        train_params, _, _ = get_params_convertors_ctx_param_name_from_function(
            self.train
        )
        train_command = click.Command(
            name="train",
            callback=self.train,
            params=train_params,
        )
        typer_click_object.add_command(train_command, "train")

        show_batch_params, _, _ = get_params_convertors_ctx_param_name_from_function(
            self.show_batch
        )
        command = click.Command(
            name="show-batch",
            callback=self.show_batch,
            params=show_batch_params,
        )
        typer_click_object.add_command(command, "show-batch")

        params, _, _ = get_params_convertors_ctx_param_name_from_function(self.tune)
        tuning_params = self.tuning_params()
        for param in params:
            if param.name in tuning_params:
                param.default = None
        command = click.Command(
            name="tune",
            callback=self.tune,
            params=params,
        )
        typer_click_object.add_command(command, "tune")

        params, _, _ = get_params_convertors_ctx_param_name_from_function(self.__call__)
        command = click.Command(
            name="predict",
            callback=self.__call__,
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
        raise NotImplementedError

    def model(self) -> nn.Module:
        raise NotImplementedError

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
            console.print(
                "Setting floating-point precision of learner to 16 bit", style="red"
            )
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

    def monitor(self):
        return None

    def goal(self):
        """
        Sets the optimality direction when evaluating the metric from `monitor`.

        By default it is set to "minimize" if the monitor metric has the string 'loss' or 'err' otherwise it is "maximize".
        """
        monitor = self.monitor()
        # Compare fastai.callback.tracker
        return "minimize" if "loss" in monitor or "err" in monitor else "maximize"

    def callbacks(self) -> list:
        """
        The list of callbacks to use with this app in the fastai training loop.

        Returns:
            list: The list of callbacks.
        """
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
        epochs: int = Param(default=20, help="The number of epochs."),
        lr_max: float = Param(default=1e-4, help="The max learning rate."),
        distributed: bool = Param(default=False, help="If the learner is distributed."),
        run_name: str = Param(
            default="",
            help="The name for this run for logging purposes. If no name is given then the name of the output directory is used.",
        ),
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
        self.init_run(run_name=run_name, output_dir=output_dir, **kwargs)

        dataloaders = run_callback(self.dataloaders, kwargs)

        learner = self.learner(dataloaders, output_dir=output_dir, **kwargs)

        with learner.distrib_ctx() if distributed == True else nullcontext():
            learner.fit_one_cycle(epochs, lr_max=lr_max, cbs=self.callbacks())

        self.save_model(learner, run_name)
        return learner

    def project_name(self) -> str:
        """
        The name to use for a project for logging purposes.

        The default is to use the class name.
        """
        return self.__class__.__name__

    def tune(
        self,
        id: str = None,
        name: str = None,
        method: str = "random",  # Should be enum
        runs: int = 1,
        min_iter: int = None,
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
                "name": name,
                "method": method,
                "parameters": parameters_config,
            }
            if self.monitor():
                sweep_config["metric"] = dict(name=self.monitor(), goal=self.goal())

            if min_iter:
                sweep_config["early_terminate"] = dict(
                    type="hyperband", min_iter=min_iter
                )

            console.print("Configuration for hyper-parameter tuning:", style="bold red")
            pprint(sweep_config)

    def init_run(self, run_name, output_dir, **kwargs):
        if not run_name:
            run_name = Path(output_dir).name
        console.print(f"from {self.project_name()}")
        console.print(f"running {run_name}")
        console.print(f"with these parameters: \n {kwargs}")

    def log(self, param):
        console.print(param)

    def logging_callbacks(self, callbacks):
        return callbacks

    def save_model(self, learner, run_name):
        learner.save(run_name)
