from fastapp.testing.testcases import FastAppTestCase
from fastapp.examples.logistic_regression import LogisticRegressionApp


class TestLogisticRegressionApp(FastAppTestCase):
    app_class = LogisticRegressionApp
