import unittest
from torch import nn

from fastapp.examples.iris import IrisApp


class TestIris(unittest.TestCase):
    def setUp(self):
        self.app = IrisApp()

    def test_model(self):
        model = self.app.model()
        # self.assertIsInstance( model, nn.Module )
        self.assertTrue(model is None)
