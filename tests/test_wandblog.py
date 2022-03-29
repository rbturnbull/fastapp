import unittest
import sys, os, shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapp.logging import WandbMixin
import wandb
from fastapp.examples.logistic_regression import LogisticRegressionApp

ARTIFACT_PATH = 'tmp/artifact'
TEST_CONFIG = {'bs':5, 'epochs':7}
def mock_init_offline(dir, project,reinit, config, **kwargs):
    # kwargs['mode'] = 'disabled'
    # run = wandb.init(dir, project, reinit, config, **kwargs
    # )
    # assert run == wandb.run
    return dict(dir=dir, project=project, reinit=reinit, config=config, kwargs = kwargs)

def mock_artifact_add_file(artifact_path):
    assert artifact_path == ARTIFACT_PATH

class TestApp(WandbMixin, LogisticRegressionApp):
    
    def __init__(self):
        super().__init__()

class WandbMixinTest(unittest.TestCase):

    def setUp(self) -> None:
        self.wandb_app =  TestApp()
        self.test_config = TEST_CONFIG.copy()
        self.wandb_path = Path('wandb_test')

        if self.wandb_path.exists():
            shutil.rmtree(self.wandb_path)
        self.wandb_path.mkdir(exist_ok=True)
        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(self.wandb_path)
        return super().tearDown()

    
    def test_init(self):
        

        self.wandb_app.init_run('wandb_test', 
        upload_model=False, mode='disabled',
        config = self.test_config
        )
        assert self.wandb_app.run is wandb.run
        # print(self.wandb_app.run.project_name)
        # print(self.wandb_app.project_name())
        # assert self.wandb_app.run.project_name == 'test_run' 

        # assert self.wandb_app.run.config['bs'] == 5
        # assert self.wandb_app.run.config['epochs'] == 7
    @patch('fastapp.logging.wandb.init',
    mock_init_offline)
    def test_init_params(self):
        self.wandb_app.init_run('wandb_test', 
        project_name='test_run2', 
        upload_model=True, mode='offline',
        config = self.test_config
        )
        # assert self.wandb_app.run[0] is wandb.run

        self.assertEqual( self.wandb_app.run['project'] , 'test_run2' )

        self.assertEqual( self.wandb_app.run['config']['bs'] , 5)
        self.assertEqual( self.wandb_app.run['config']['epochs'] , 7)
        self.assertEqual( self.wandb_app.upload_model , True)
        self.assertEqual( self.wandb_app.run['kwargs']['mode'], 'offline')


    def test_log(self):

        self.test_init()
        self.wandb_app.log({'test_param':'test_out'})

    def test_log_artifact(self):

        self.test_init()

    def test_save_model(self):
        pass


    def test_logging_callbacks(self):
        pass
