import yaml
from pathlib import Path
import unittest
from torch import nn
import importlib
import pdb
import sys

from ..apps import FastApp


class FastAppTestCase:
    app_class = None
    expected_base = None

    @classmethod
    def setUpClass(cls):
        if cls is FastAppTestCase:
            raise unittest.SkipTest("Skip FastAppTestCase since it is a base class")
        super().setUpClass()

    def setUp(self):
        self.app = self.get_app()
        self.expected_dir = self.get_expected_dir()

    def get_expected_base(self) -> Path:
        if not self.expected_base:
            module = importlib.import_module(self.__module__)
            self.expected_base = Path(module.__file__).parent / "expected"

        self.expected_base = Path(self.expected_base)
        return self.expected_base

    def get_expected_dir(self) -> Path:
        """
        Returns the path to the directory where the expected files.

        It creates the directory if it doesn't already exist.
        """
        expected_dir = self.get_expected_base() / self.__class__.__name__
        expected_dir.mkdir(exist_ok=True, parents=True)
        return expected_dir

    def get_app(self) -> FastApp:
        """
        Returns an instance of the app for this test case.

        It instantiates an object from `app_class`.
        Override `app_class` or this method so the correct app is returned from calling this method.
        """
        # pdb.set_trace()
        self.assertIsNotNone(self.app_class)
        app = self.app_class()

        self.assertIsInstance(app, FastApp)
        return app

    def subtests(self, name: str):
        if name.startswith("test_"):
            name = name[5:]
        directory = self.expected_dir / name
        directory.mkdir(exist_ok=True, parents=True)
        files = list(directory.glob("*.yaml"))

        if len(files) == 0:
            raise unittest.SkipTest(
                f"Skipping '{name}' because there are no files with expected output in '{directory}'."
            )

        for file in files:
            with open(file) as f:
                file_dict = yaml.safe_load(f)
                params = file_dict.get("params", {})
                output = file_dict.get("output", "")
                with self.subTest(msg=file.name, **params):
                    yield params, output

    def test_model(self):
        for params, output in self.subtests(sys._getframe().f_code.co_name):
            model = self.app.model(**params)
            if not output:
                self.assertIsNone(model)
            else:
                self.assertIsInstance(model, nn.Module)
                self.assertEqual(str(model), output)
