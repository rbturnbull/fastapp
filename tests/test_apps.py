import fastapp as fa
from fastapp.apps import FastAppInitializationError
import pytest


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
