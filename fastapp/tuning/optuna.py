from pathlib import Path

try:
    import optuna
    from optuna.integration import FastAIV2PruningCallback
    from optuna import samplers
except:
    raise Exception(
        "No module named 'optuna'. Please install this as an extra dependency or choose a different optimization engine."
    )

from ..util import call_func


def get_sampler(method):
    method = method.lower()
    if method.startswith("tpe"):
        sampler = samplers.TPESampler()
    elif method.startswith("cma"):
        sampler = samplers.CmaEsSampler()
    elif method.startswith("grid"):
        sampler = samplers.GridSampler()
    elif method.startswith("random"):
        sampler = samplers.RandomSampler()
    else:
        raise Exception(f"Cannot interpret sampling method '{method}' using Optuna.")
    return sampler


def suggest(trial, name, param):
    if param.tune_choices:
        return trial.suggest_categorical(name, param.tune_choices)
    elif param.annotation == float:
        return trial.suggest_float(name, param.tune_min, param.tune_max, log=param.log)
    elif param.annotation == int:
        return trial.suggest_int(name, param.tune_min, param.tune_max, log=param.log)

    raise NotImplementedError("Optuna Tuning Engine cannot understand param '{name}': {param}")


def optuna_tune(
    app,
    storage: str = "",
    name: str = None,
    method: str = "random",  # Should be enum
    runs: int = 1,
    **kwargs,
):
    def objective(trial: optuna.Trial):
        run_kwargs = dict(kwargs)

        tuning_params = app.tuning_params()

        for key, value in tuning_params.items():
            if key not in kwargs or kwargs[key] is None:
                run_kwargs[key] = suggest(trial, key, value)

        trial_name = f"trial-{trial.number}"

        output_dir = Path(run_kwargs.get("output_dir", "."))
        run_kwargs["output_dir"] = output_dir / trial.study.study_name / trial_name
        run_kwargs["project_name"] = trial.study.study_name
        run_kwargs["run_name"] = trial_name

        # Train
        learner = call_func(app.train, **run_kwargs)

        # Return metric from recorder
        # The slice is there because 'epoch' is prepended to the list but it isn't included in the values
        metric_index = learner.recorder.metric_names[1:].index(app.monitor())
        metric_values = map(lambda row: row[metric_index], learner.recorder.values)
        metric_function = min if app.goal()[:3] == "min" else max
        metric_value = metric_function(metric_values)
        return metric_value

    study = optuna.create_study(
        study_name=name, storage=storage, sampler=get_sampler(method), load_if_exists=True, direction=app.goal()
    )
    study.optimize(objective, n_trials=runs)
