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

from fastcore.meta import delegates
from fastapp.params import Param
from fastapp.apps import run_callback
import mlflow
import matplotlib
# import plotly
import pickle
from mlflow.tracking import MlflowClient  


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


class WandbMixin(object):
    """app logging mixin for logging to weights and biases

    Args:
        object (WandbMixin): mixin for logging training runs, params, and metrics to Weights and Biases
    """

    def __init__(self):
        super().__init__()
        delegates(to=self.init_run)(self.train)

    def init_run(
        self,
        run_name: str,
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

        Args:
            run_name (str): specify the name of the run
            output_dir (Union[Path, str]): output directory of model and other artifacts
            project_name (Optional[str], optional): name of project. Defaults to None, and uses name of App class
            config (dict, optional):dictionary of config params to log for the wandb run. Defaults to {}.
            upload_model (Union[Param, bool], optional): If true, will upload model to Weights and Biases, else, it logs a file reference to output_dir. 
            Defaults to Param( default=False, help="If true, logs model to WandB project" ).
            kwargs: additional kwargs for 'wandb.init'
        """

        self.upload_model = upload_model

        if project_name is None:
            project_name = self.project_name()
        self.run = wandb.init(
            dir=output_dir, project=project_name, reinit=True, config=config, **kwargs
        )

    def log(self, param: dict):
        """log a dictionary of metrics to weights and biases, parameters should be logged in the config with init run

        Args:
            param (dict): dictionary of metrics
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

        Args:
            artifact_path (Union[Path, str]): path to file to be uploaded
            artifact_name (str): artifact name in weights and biases project
            artifact_type (str): type of artifact
            upload (bool, optional): if True, uploads file to Weights and Biases web app. 
            If False, references the path to the object and tracks changes. Defaults to False.
        """

        artifact = wandb.Artifact(artifact_name, type=artifact_type, **kwargs)
        if upload == True:
            artifact.add_file(artifact_path)
        else:
            artifact.add_reference(artifact_path)
        self.run.log_artifact(artifact)

    def logging_callbacks(self, callbacks: list):
        """function that adds weights and biases callbacks to callback list before training

        Args:
            callbacks (list): list of pregenerated fastai callbacks

        Returns:
            _type_: updated list of fastai callbacks including weights and biases logging callbacks

        """

        wandb_callback = WandbCallback(log_preds=False)
        callbacks.extend([wandb_callback, WandbCallbackTime(wandb_callback=wandb_callback)])
        return callbacks

    def save_model(self, learner: Learner, run_name: str):
        """Saves the model after training, and logs it as an artifact or file reference.

        Args:
            learner (Learner): fastai learner containing model weights
            run_name (str): name of the run to save the model to

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

        Args:
            id (str, optional): sweep ID, only necessary if sweep has already been generated for the project. Defaults to None.
            name (str, optional): name of the sweep run, defaullts to project name. Defaults to None.
            method (str, optional): hyperparameter sweep method, can be random for random, grid for grid search, 
            and bayes for bayes optimisation. Defaults to "random".
            min_iter (int, optional): minimun number of iterations. Defaults to None.

        Returns:
            str: sweep id
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
                sweep_config["early_terminate"] = dict(type="hyperband", min_iter=min_iter)
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

 


"""The logging module contains prebuilt mixin classes
for logging model runs to different mlOps metadata and artifact stores.
At minimum, these classes need to contain the following methods:
`init_run` - this initialises a run in the mlOps framework for the training run to log too
`log` - function for logging a parameter or metric. Overwrites a function that just prints in to stdout
`log_artifact`- a function to log file artifacts (files and models etc) to the mlOps artifact store
`save_model` - function that will save the model weights and log them to an artifact store, overwrites the learner.save function in the base app.
Other optional methods that may be added updated:
Currently, a mixin for using MLFlow implemented, and includes
a tuning function for using MLFlow sweeps for hyperparameter tuning
"""

install()
console = Console()

def assert_dir_exists(path:Union[str, Path]):
    """check if the directory_path exsits and if not, make directory

    Args:
        path (Union[str, Path]): specified path (if it doesn't exist, make one)
    """
    if isinstance(path, (str)):
        path = Path(path)
    if path.exists() == False:
        path.mkdir()
        
def _experiment_exists(experiment_id):
    """check if experiment_id exists

    Args:
        experiment_id (_type_): experiment_id to be examined
Optional[str]
    Returns:
        boolean, str: True if experiment_id or name exists, then return id/name. Otherwise, return False
    """
    experiments = mlflow.list_experiments()
    if experiment_id in [e.experiment_id for e in experiments]:
        return True, 'id'
    
    elif experiment_id in [e.name for e in experiments]:
        return True, 'name'
    else:
        return False, ''
def create_experiment(experiment_name: Optional[str] = None):
    """if the experiment does not exist, create an experiment with the specified name

    Args:
        experiment_name (Optional[str], optional): name of the experiment. Defaults to None.

    Returns:
        mlflow.experiment: instance of an mlflow experiment named experiment_name
    """
    if _experiment_exists(experiment_name)[0] ==False:
        return mlflow.create_experiment(name = experiment_name)
    
def get_experiment_id(experiment_id):
    """fetch experiment that has experiment_id

    Args:
        experiment_id (_type_): _description_

    Returns:
        _type_: _description_
    """
    e_exists, e_type = _experiment_exists(experiment_id)
    if e_exists:
        if e_type == 'name':
            return mlflow.get_experiment_by_name(name=experiment_id).experiment_id
        else:
            return experiment_id
    else:
        return create_experiment(experiment_id)

    
class MLFlowMixin(object):
    """app logging mixin for logging to mlflow

    Args:
        object (MLFlowMixin): mixin for logging training runs, params, and metrics to Weights and Biases
    """

    def __init__(self):
        delegates(to=self.init_run)(self.train)
        super().__init__()
        

    def init_run(
        self,
        run_name: Optional[str] = None,
        output_dir: Optional[Union[Path, str]] = '', 
        experiment_id: Optional[str] = None,
        **kwargs,):
        """starts an mlflow run (with a given experiment, optional run name and tracking_uri) and
        initiates mlflow.fastai.autolog(log_models=False)

        Args:
            run_name (Optional[str], optional): name of run. Defaults to None, and uses name of App class. 
            output_dir (Optional[Union[Path, str]], optional): output directory of model and other artifacts. Defaults to ''.
            experiment_id (Optional[str], optional): ID of the experiment under which to create the current run. Defaults to None, and uses name of App class.

        """
  
        #         self.upload_model = upload_model
        # optional 'tracking_uri' can be set
        if len(str(output_dir))> 0:
            assert_dir_exists(output_dir)
            mlflow.set_tracking_uri(f'file:./{str(output_dir)}')
        
        # if no experiment_id is given, then it should create self.project_name()
        if experiment_id is None:
            experiment_id = self.project_name()    
        experiment_id = get_experiment_id(experiment_id)
        import pdb;pdb.set_trace()

        # starts an mlflow run with a given experiment_id and an optional run_name
        if run_name is None:
            mlflow.start_run(experiment_id=experiment_id) 
        else:
            mlflow.start_run(experiment_id=experiment_id, run_name=run_name)   #run_name parameter used only when run_id is unspecified
        
        mlflow.fastai.autolog(log_models=False)
        
        #checking functions    
        self.run = mlflow.active_run()
        print("Active run_id: {}".format(self.run.info.run_id))
        tracking_uri = mlflow.get_tracking_uri()
        print("Current tracking uri: {}".format(tracking_uri))
        
        
    def log(self, param: dict, parameter_metric: bool=False, step: Optional[int] = None):
        """log (param:dict) as a set of parameters or a set of metrics
        if parameter metric = True, log as a set of metric with an optional argument 'step'


        Args:
            param (dict): dictionary of parameters to be logged
            parameter_metric (bool, optional): if True, log as a set of metricss. Defaults to False.
            step (Optional[int], optional): an optional argument. Defaults to None.
        """

        if parameter_metric == True:
            log_metrics(param, step=step)
        else:
            log_params(param)         

                 
    def log_artifact(
        self,
        artifact,
        artifact_path: Union[Path, str],
#         artifact_name: str,
#         artifact_type: str,
#         upload: bool = False,
        **kwargs,
    ):
        """Input an artifact (padans df/matplotlib/plotly.figure/dict/str/path) to a saved file

        Args:
            artifact (_type_): artifact to be logged
            artifact_path (Union[Path, str]): path to file to be uploaded
        """

#         artifacts = mlflow.artifacts.download_artifacts
    
        if isinstance(artifact, pd.DataFrame):
            csv_file = artifact.to_csv(None, sep='\t')
            mlflow.log_text(csv_file, artifact_path)

#         elif isinstance(artifact, plotly.graph_objs._figure.Figure ):
#             mlflow.log_figure(artifact, artifact_path)

        elif isinstance(artifact, matplotlib.figure.Figure):
            mlflow.log_figure(artifact, artifact_path)

        elif isinstance(artifact, dict):
            mlflow.log_dict(artifact, artifact_path)

        elif isinstance(artifact, str):
            mlflow.log_text(artifact, artifact_path)

#         elif isinstance(artifact, ):
#             mlflow.log_artifact()
        else:
            pickle.dump(artifact, open(artifact_path, 'wb'))



    def save_model(self, learner: Learner, path):
        """saves model as a pytorch model (input = fastai.learner, output= pytorch version of the model)

        Args:
            learner (Learner): fastai learner containing model weights
            path (str): path to file to be uploaded
        """


        mlflow.pytorch.log_model(learner.model, artifact_path=path)

        #         super().save_model(learner, run_name)
        # mlflow.end_run()
        #         model_path = learner.path / learner.model_dir / run_name
        #         mlflow.fastai.save_model(learner, model_path,
        #                           serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)

