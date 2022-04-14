import fastapp as fa


def test_model_defaults_change():
    class DummyApp(fa.FastApp):
        def model(self, size: int = fa.Param(default=2)):
            assert size == 2

    app = DummyApp()
    app.model()
