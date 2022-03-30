import unittest

from fastapp.testing.testcases import FastAppTestCase
from fastapp.examples.logistic_regression import LogisticRegressionApp


class TestLogisticRegressionApp(FastAppTestCase, unittest.TestCase):
    app_class = LogisticRegressionApp
