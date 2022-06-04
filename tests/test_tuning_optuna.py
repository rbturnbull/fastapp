from optuna import samplers
import pytest

from .tuning_test_app import TuningTestApp


def test_optuna_tune():
    app = TuningTestApp()
    result = app.tune(engine="optuna", runs=10, seed=42)
    assert len(result.trials) == 10
    assert result.best_value > 9.9
    assert result.best_trial.number == 8


def test_get_sampler():
    from fastapp.tuning.optuna import get_sampler

    assert isinstance(get_sampler("tpe"), samplers.TPESampler)
    assert isinstance(get_sampler("cma-es"), samplers.CmaEsSampler)
    # assert isinstance(get_sampler("grid"), samplers.GridSampler)
    assert isinstance(get_sampler("random"), samplers.RandomSampler)
    with pytest.raises(NotImplementedError):
        get_sampler("bayes")
