from optuna import samplers
import pytest
import numpy as np
from numpy.testing import assert_allclose
from .tuning_test_app import TuningTestApp


# def test_skopt_tune_default():
#     app = TuningTestApp()
#     runs = 120
#     result = app.tune(engine="skopt", runs=runs, seed=42)
#     assert len(result.trials) == runs
#     assert result.best_value > 9.93
#     assert result.best_trial.number == 104
#     assert isinstance(result.sampler, samplers.RandomSampler)
#     df = result.trials_dataframe()
#     assert "params_a" in df.columns
#     assert "params_x" in df.columns
#     assert "params_string" in df.columns


# def test_skopt_tune_cmaes():
#     app = TuningTestApp()
#     result = app.tune(engine="skopt", method="cmaes", runs=15, seed=42, string="abcdefghij")
#     assert len(result.trials) == 15
#     assert result.best_value > 9.8
#     assert result.best_trial.number == 6
#     assert isinstance(result.sampler, samplers.CmaEsSampler)
#     df = result.trials_dataframe()
#     assert "params_a" in df.columns
#     assert "params_x" in df.columns
#     assert "params_string" not in df.columns


def test_skopt_tune_random():
    app = TuningTestApp()
    runs = 100
    result = app.tune(engine="skopt", method="random", runs=runs, seed=42)
    assert len(result.func_vals) == runs
    assert result.fun < -9.9
    assert result.space.n_dims == 3
    assert type(result.space[0][1]).__name__ == 'Real'
    assert type(result.space[1][1]).__name__ == 'Integer'
    assert type(result.space[2][1]).__name__ == 'Categorical'
    for found, desired in zip(result.x, [1.9370031589297412, 10, 'abcdefghij']):
        if isinstance(found, float):
            assert_allclose(found, desired)
        else:
            assert found == desired


def test_get_optimizer():
    import skopt
    from fastapp.tuning.skopt import get_optimizer

    assert get_optimizer("bayes") == skopt.gp_minimize
    assert get_optimizer("gp") == skopt.gp_minimize
    assert get_optimizer("forest") == skopt.forest_minimize
    assert get_optimizer("random") == skopt.dummy_minimize
    assert get_optimizer("gbrt") == skopt.gbrt_minimize
    assert get_optimizer("gradientboost") == skopt.gbrt_minimize
    with pytest.raises(NotImplementedError):
        get_optimizer("tpe")
    with pytest.raises(NotImplementedError):
        get_optimizer("cma")
