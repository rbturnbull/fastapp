from pathlib import Path
from unittest.mock import patch
from fastai.learner import Learner
import pytest
from click.testing import CliRunner
import fastapp as fa
from fastapp.apps import FastAppInitializationError


def test_model_defaults_change():
    class DummyApp(fa.FastApp):
        def model(self, size: int = fa.Param(default=2)):
            assert size == 2

    DummyApp().model()


def test_model_unimplemented_error():
    with pytest.raises(NotImplementedError):
        fa.FastApp().model()


def test_dataloaders_unimplemented_error():
    with pytest.raises(NotImplementedError):
        fa.FastApp().dataloaders()


def test_assert_initialized():
    class DummyApp(fa.FastApp):
        def __init__(self):
            pass

    with pytest.raises(FastAppInitializationError):
        DummyApp().cli()


def test_click():
    cli = fa.FastApp.click()
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert '[OPTIONS] COMMAND [ARGS]' in result.output


def test_str():
    class DummyApp(fa.FastApp):
        pass

    app = DummyApp()
    assert str(app) == "DummyApp"


def test_pretrained_local_path_default():
    with pytest.raises(FileNotFoundError):
        fa.FastApp().pretrained_local_path()


@patch.object(Path, 'is_file', lambda x: True)
def test_pretrained_local_path():
    class DummyApp(fa.FastApp):
        def pretrained_location(self):
            return "model.h5"

    app = DummyApp()
    path = app.pretrained_local_path()

    assert path.is_absolute()
    assert path.name == "model.h5"
    assert path.parent.name == "tests"


@patch.object(Path, 'is_file', lambda x: True)
def test_pretrained_local_path_override():
    path = fa.FastApp().pretrained_local_path("model2.h5")
    assert path.is_absolute()
    assert path.name == "model2.h5"
    assert path.parent == Path.cwd()


def test_pretrained_local_path_not_found():
    with pytest.raises(FileNotFoundError):
        fa.FastApp().pretrained_local_path("model3.h5")


def test_build_learner_func():
    assert Learner in fa.FastApp().build_learner_func().__mro__


def test_goal_empty():
    class DummyApp(fa.FastApp):
        def monitor(self):
            return None

    app = DummyApp()
    assert not app.goal()
