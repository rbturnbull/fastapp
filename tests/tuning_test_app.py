import fastapp as fa

MOCK_METRIC = "metric"


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
        **kwargs,
    ):
        assert isinstance(x, float)
        assert -10.0 <= x <= 10.0

        value = 10.0 - (x - 2.0) ** 2
        return MockLearner(value=value)
