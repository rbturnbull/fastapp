import unittest
from fastapp.testing.testcases import FastAppTestCase
from fastapp.examples.iris import IrisApp


class TestIris(FastAppTestCase, unittest.TestCase):
    app_class = IrisApp
