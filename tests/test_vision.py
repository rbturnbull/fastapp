import pytest
from fastapp.testing import FastAppTestCase
from fastapp.vision import VisionApp


class TestVisionApp(FastAppTestCase):
    app_class = VisionApp

    def test_model_incorrect(self):
        app = self.get_app()
        with pytest.raises(ValueError):
            app.model(model_name="resnet1000")
