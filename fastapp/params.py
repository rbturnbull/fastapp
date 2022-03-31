import math
from typer.models import OptionInfo


class Param(OptionInfo):
    def __init__(
        self,
        default = None,
        tune=False,
        tune_min=None,
        tune_max=None,
        log=False,
        distribution=None,
        annotation=None,
        **kwargs,
    ):
        super().__init__(default = default, **kwargs)
        self.tune = tune
        self.log = log
        self.tune_min = tune_min if tune_min is not None else self.min
        self.tune_max = tune_max if tune_max is not None else self.max
        self.annotation = annotation
        self.distribution = distribution
        if distribution:
            raise NotImplementedError("Distribution for parameters not implemented yet")

    def config(self):
        if self.annotation in [int, float]:
            assert self.tune_min is not None
            assert self.tune_max is not None

            distribution = "log_uniform" if self.log else "uniform"
            if self.annotation == int:
                distribution = f"q_{distribution}"
            if self.log:
                return dict(
                    distribution=distribution,
                    min=math.log(self.tune_min),
                    max=math.log(self.tune_max),
                )

            return dict(
                distribution=distribution,
                min=self.tune_min,
                max=self.tune_max,
            )

        # TODO add categorical

        import pdb

        pdb.set_trace()
        raise NotImplementedError
