import fastapp as fa
import numpy as np

MOCK_METRIC = "metric"

categorical_choices = ["abcdefghij", "baby", "c"]


class MockRecorder:
    def __init__(self, value, metric_name=MOCK_METRIC, epochs=20):
        self.metric_names = ["epoch", "other", metric_name]
        self.values = []
        for epoch in range(epochs):
            self.values.append([-2.22, epoch - epochs + 1 + value])


class MockLearner:
    def __init__(self, value):
        self.recorder = MockRecorder(value=value)


class TuningTestApp(fa.FastApp):
    def monitor(self):
        return MOCK_METRIC

    def train(
        self,
        x: float = fa.Param(default=0.0, tune=True, min=-10.0, max=10.0, help="A real parameter in [-10.0,10.0]."),
        a: int = fa.Param(default=2, tune=True, min=1, max=12, help="An integer parameter in [1,12]."),
        string: str = fa.Param(
            default="baby",
            tune=True,
            tune_choices=categorical_choices,
            help="An string parameter which is either 'abcdefghij', 'baby', or 'c'.",
        ),
        **kwargs,
    ):
        assert isinstance(x, float)
        assert -10.0 <= x <= 10.0
        assert isinstance(a, int) or isinstance(a, np.integer)
        assert 1 <= a <= 12
        assert isinstance(string, str)
        assert string in categorical_choices

        c = len(string)

        value = c - a * (x - 2.0) ** 2
        return MockLearner(value=value)
