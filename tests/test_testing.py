from unittest.mock import patch
from fastapp.examples.iris import IrisApp
from fastapp.testing import FastAppTestCase


def get_test_case():
    class DummyApp(IrisApp):
        pass

    class TestDummyApp(FastAppTestCase):
        app_class = DummyApp

    return TestDummyApp()


@patch('builtins.input', lambda _: 'y')
def test_model_interactive():
    test_case = get_test_case()
    test_case.test_model(interactive=True)
