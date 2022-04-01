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

install()
console = Console()


class WandbMixin(object):
    """app logging mixin for logging to weights and biases

    :param object: mixin for logging training runs, params, and metrics to Weights and Biases
    :type object: WandbMixin
    """

    def __init__(self):
        super().__init__()
        delegates(to=self.init_run)(self.train)

    def init_run(
        self,
        output_dir: Union[Path, str],
        project_name: Optional[str] = None,
        config: dict = {},
        upload_model: Union[Param, bool] = Param(
            default=False, help="If true, logs model to WandB project"
        ),
        **kwargs,
    ):
        """initialises a weights and biases run for each training run or sweep, stores the
        current run as the `self.run` attribute in the app instance.

        :param output_dir: output directory of model and other artifacts
        :type output_dir: Union[Path, str]
        :param project_name: name of project, defaults to None, and uses name of App class
        :type project_name: Optional[str], optional
        :param config: dictionary of config params to log for the wandb run, defaults to {}
        :type config: dict, optional
        :param upload_model: If true, will upload model to Weights and Biases, else, it logs a file reference to output_dir, defaults to Param( default=False, help="If true, logs model to WandB project" )
        :type upload_model: Union[Param, bool], optional
        :param kwargs: additional kwargs for `wandb.init`
        """
        self.upload_model = upload_model

        if project_name is None:
            project_name = self.project_name()
        self.run = wandb.init(
            dir=output_dir, project=project_name, reinit=True, config=config, **kwargs
        )

    def log(self, param: dict):
        """log a dictionary of metrics to weights and biases, parameters should be logged in the config with init run

        :param param: dictionary of metrics
        :type param: dict
        """
        wandb.log(param)

    def log_artifact(
        self,
        artifact_path: Union[Path, str],
        artifact_name: str,
        artifact_type: str,
        upload: bool = False,
        **kwargs,
    ):
        """Logs a file artifact to weights and biases run. Can either upload the file, or
        reference the file depending on upload parameter.

        :param artifact_path: path to file to be uploaded
        :type artifact_path: Union[Path, str]
        :param artifact_name: artifact name in weights and biases project
        :type artifact_name: str
        :param artifact_type: type of artifact
        :type artifact_type: str
        :param upload: if True, uploads file to Weights and Biases web app, if False, references the path to the object and tracks changes, defaults to False
        :type upload: bool, optional
        """
        artifact = wandb.Artifact(artifact_name, type=artifact_type, **kwargs)
        if upload == True:
            artifact.add_file(artifact_path)
        else:
            artifact.add_reference(artifact_path)
        self.run.log_artifact(artifact)

    def logging_callbacks(self, callbacks: list):
        """function that adds weights and biases callbacks to callback list before training

        :param callbacks: list of pregenerated fastai callbacks
        :type callbacks: list
        :return: updated list of fastai callbacks including weights and biases logging callbacks
        :rtype: _type_
        """
        wandb_callback = WandbCallback(log_preds=False)
        callbacks.extend(
            [wandb_callback, WandbCallbackTime(wandb_callback=wandb_callback)]
        )
        return callbacks

    def save_model(self, learner: Learner, run_name: str):
        """Saves the model after training, and logs it as an artifact or file reference.

        :param learner: fastai learner containing model weights
        :type learner: Learner
        :param run_name: name of the run to save the model to
        :type run_name: str
        """
        super().save_model(learner, run_name)

        model_path = learner.path / learner.model_dir / run_name
        # import pdb;pdb.set_trace()
        self.log_artifact(model_path, run_name, "model", upload=self.upload_model)

    def tune(
        self,
        id: str = None,
        name: str = None,
        method: str = "random",  # Should be enum
        runs: int = 1,
        min_iter: int = None,
        **kwargs,
    ) -> str:
        """This initiates hyperparameter tuning using weights and biases sweeps

        :param id: sweep ID, only necessary if sweep has already been generated for the project, defaults to None
        :type id: str, optional
        :param name: name of the sweep run, defaullts to project name, defaults to None
        :type name: str, optional
        :param method: hyperparameter sweep method, can be random for random, grid for grid search, and bayes for bayes optimisation defaults to "random"
        :type method: str, optional
        :param min_iter: minimun number of iterations, defaults to None
        :type min_iter: int, optional
        :return: sweep id
        :rtype: str
        """
        if not name:
            name = f"{self.project_name()}-tuning"
        # self.init_run(run_name=name)
        if not id:

            parameters_config = dict()
            tuning_params = self.tuning_params()

            for key, value in tuning_params.items():
                if ((key in kwargs) and (kwargs[key] is None)) or (key not in kwargs):
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

            id = wandb.sweep(sweep_config, project=name)
            console.print(f"The wandb sweep id is: {id}", style="bold red")

        def agent_train():
            with wandb.init() as run:
                run_kwargs = dict(kwargs)
                run_kwargs.update(wandb.config)
                if "output_dir" in run_kwargs:
                    run_kwargs["output_dir"] = Path(run_kwargs["output_dir"]) / run.name

                console.print("Training with parameters:", style="bold red")
                pprint(run_kwargs)

                run_callback(self.train, run_kwargs)

        wandb.agent(id, function=agent_train, count=runs, project=name)

        return id
