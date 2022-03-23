import unittest
from torch import nn

from fastapp.examples.iris import IrisApp

class TestIris(unittest.TestCase):

    def setUp(self):
        self.app = IrisApp()

    def test_model(self):
        self.assertIsInstance( self.app.model(), nn.Module )
