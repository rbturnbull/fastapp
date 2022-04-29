from pathlib import Path
from unittest.mock import patch
from fastapp.callbacks.wandb import FastAppWandbCallback, wandb
from fastapp.examples.iris import IrisApp
import tempfile


def test_default_no_wandb():
    app = IrisApp()
    callbacks = app.callbacks()

    callback = None
    for c in callbacks:
        if isinstance(c, FastAppWandbCallback):
            callback = c
            break
    assert callback is None
    assert wandb.run is None


def test_wandb_init():
    app = IrisApp()
    callbacks = app.callbacks(wandb=True, wandb_mode="disabled")

    callback = None
    for c in callbacks:
        if isinstance(c, FastAppWandbCallback):
            callback = c
            break
    assert callback is not None
    assert wandb.run is not None
    assert callback.run is wandb.run


def test_wandb_after_epoch():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = IrisApp()
        app.train(wandb=True, wandb_mode="offline", wandb_dir=tmpdir, epochs=1)
        assert isinstance(wandb.summary['time'], float)
        wandb.finish()  # needs to be called before deleting tmpdir
