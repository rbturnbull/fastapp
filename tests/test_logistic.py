import unittest
from torch import nn

from fastapp.examples.logistic_regression import LogisticRegressionApp

class TestLogisticRegressionApp(unittest.TestCase):

    def setUp(self):
        self.app = LogisticRegressionApp()

    def test_model(self):
        model = self.app.model()
        self.assertIsInstance( model, nn.Module )
