import pytest
import math
from fastapp import params


def test_int():
    param = params.Param(annotation=int, tune_min=0, tune_max=100)
    config = param.config()

    assert isinstance(config, params.ParamConfig)
    assert config.min == 0
    assert config.max == 100
    assert config.distribution == "q_uniform"


def test_int_log():
    param = params.Param(annotation=int, log=True, tune_min=1, tune_max=100)
    config = param.config()

    assert isinstance(config, params.ParamConfig)
    assert config.min == math.log(1)
    assert config.max == math.log(100)
    assert config.distribution == "q_log_uniform"


def test_float_log():
    param = params.Param(annotation=float, log=True, tune_min=1.0, tune_max=100.0)
    config = param.config()

    assert isinstance(config, params.ParamConfig)
    assert config.min == math.log(1)
    assert config.max == math.log(100)
    assert config.distribution == "log_uniform"


def test_float():
    param = params.Param(annotation=float, tune_min=1.0, tune_max=100.0)
    config = param.config()

    assert isinstance(config, params.ParamConfig)
    assert config.min == 1.0
    assert config.max == 100.0
    assert config.distribution == "uniform"


def test_assert_distribution():
    with pytest.raises(NotImplementedError):
        params.Param(annotation=float, tune_min=1.0, tune_max=100.0, distribution="uniform")
