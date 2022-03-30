from fastapp.testing.testcases import FastAppTestCase
from fastapp.examples.iris import IrisApp


class TestIris(FastAppTestCase):
    app_class = IrisApp
