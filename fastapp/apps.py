from contextlib import nullcontext
from pathlib import Path
from types import MethodType
from collections.abc import Iterable
import inspect
from typing import List, Optional, Union, Dict
from typing import get_type_hints
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
from typer.models import OptionInfo
from rich.pretty import pprint
from rich.console import Console
from rich.traceback import install

install()
console = Console()

from .params import Param


import types


class FastAppInitializationError(Exception):
    pass


def copy_func(f, name=None):
    """
    Returns a deep copy of a function.

    The new function has the same code, globals, defaults, closure, annotations, and name
    (unless a new name is provided)

    Derived from https://stackoverflow.com/a/30714299
    """
    fn = types.FunctionType(f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__)
    # in case f was given attrs (note this dict is a shallow copy):
    fn.__dict__.update(f.__dict__)
    fn.__annotations__.update(f.__annotations__)
    return fn


def run_callback(callback, params):
    allowed_params, _, _ = get_params_convertors_ctx_param_name_from_function(callback)
    allowed_param_names = [p.name for p in allowed_params]
    kwargs = {key: value for key, value in params.items() if key in allowed_param_names}
    return callback(**kwargs)


def change_typer_to_defaults(func):
    func = getattr(func, "__func__", func)
    # signature = inspect.signature(func)

    # # Create a dictionary with both the existing parameters for the function and the new ones
    # parameters = dict(signature.parameters)

    # for key, value in parameters.items():
    #     if isinstance(value.default, OptionInfo):
    #         parameters[key] = value.replace(default=value.default.default)

    # func.__signature__ = signature.replace(parameters=parameters.values())

    # import pdb; pdb.set_trace()

    # Change defaults directly
    if func.__defaults__ is not None:
        func.__defaults__ = tuple(
            [value.default if isinstance(value, OptionInfo) else value for value in func.__defaults__]
        )


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


def add_kwargs(to_func, from_funcs):
    """Adds all the keyword arguments from one function to the signature of another function.

    Args:
        from_funcs (callable or iterable): The function with new parameters to add.
        to_func (callable): The function which will receive the new parameters in its signature.
    """

    if not isinstance(from_funcs, Iterable):
        from_funcs = [from_funcs]

    for from_func in from_funcs:
        # Get the existing parameters
        to_func = getattr(to_func, "__func__", to_func)
        from_func = getattr(from_func, "__func__", from_func)
        from_func_signature = inspect.signature(from_func)
        to_func_signature = inspect.signature(to_func)

        # Create a dictionary with both the existing parameters for the function and the new ones
        to_func_parameters = dict(to_func_signature.parameters)

        if "kwargs" in to_func_parameters:
            kwargs_parameter = to_func_parameters.pop("kwargs")

        from_func_kwargs = {
            k: v
            for k, v in from_func_signature.parameters.items()
            if v.default != inspect.Parameter.empty and k not in to_func_parameters
        }
        # to_func_parameters['kwargs'] = kwargs_parameter

        to_func_parameters.update(from_func_kwargs)

        # Modify function signature with the parameters in this dictionary
        # print('to_func', hex(id(to_func)))
        to_func.__signature__ = to_func_signature.replace(parameters=to_func_parameters.values())


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

        # Add keyword arguments to the signatures of the methods used in the CLI
        add_kwargs(to_func=self.train, from_funcs=[self.dataloaders, self.model])
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
                callback=version_callback,
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
        self.init_run(run_name=run_name, output_dir=output_dir)

        dataloaders = run_callback(self.dataloaders, kwargs)

        # Allow the dataloaders to go to GPU so long as it hasn't explicitly been set as a different device
        if dataloaders.device is None:
            dataloaders.cuda()  # This will revert to CPU if cuda is not available

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
                if ((key in kwargs) and (kwargs[key] is None)) or (key in kwargs):
                    parameters_config[key] = value.config()

            sweep_config = {
                "name": name,
                "method": method,
                "parameters": parameters_config,
            }
            if self.monitor():
                sweep_config["metric"] = dict(name=self.monitor(), goal=self.goal())

            if min_iter:
                sweep_config["early_terminate"] = dict(type="hyperband", min_iter=min_iter)

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
