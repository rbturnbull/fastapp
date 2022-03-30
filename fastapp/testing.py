import sys
import yaml
import importlib
import pytest
from pathlib import Path

from torch import nn
from collections import OrderedDict

from .apps import FastApp

######################################################################
## pytest fixtures
######################################################################


def pytest_addoption(parser):
    parser.addoption(
        "--prompt",
        action="store_true",
        help="Whether or not to prompt for saving output in new expected files.",
    )


@pytest.fixture
def prompt_option(request):
    return request.config.getoption("--prompt")


######################################################################
## YAML functions from https://stackoverflow.com/a/8641732
######################################################################
class quoted(str):
    pass


def quoted_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


yaml.add_representer(quoted, quoted_presenter)


class literal(str):
    pass


def literal_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(literal, literal_presenter)


def ordered_dict_presenter(dumper, data):
    return dumper.represent_dict(data.items())


yaml.add_representer(OrderedDict, ordered_dict_presenter)

######################################################################
## FastApp Testing Utils
######################################################################


class FastAppTestCase:
    app_class = None
    expected_base = None

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
        assert self.app_class is not None
        app = self.app_class()

        assert isinstance(app, FastApp)
        return app

    def subtests(self, name: str):
        if name.startswith("test_"):
            name = name[5:]
        directory = self.get_expected_dir() / name
        directory.mkdir(exist_ok=True, parents=True)
        files = list(directory.glob("*.yaml"))

        assert len(files) >= 0

        for file in files:
            with open(file) as f:
                file_dict = yaml.safe_load(f) or {}
                params = file_dict.get("params", {})
                output = file_dict.get("output", "")
                yield params, output, file

    def test_model(self, prompt_option):
        app = self.get_app()
        for params, output, file in self.subtests(sys._getframe().f_code.co_name):
            model = app.model(**params)
            if model is None:
                model_summary = "None"
            else:
                assert isinstance(model, nn.Module)
                model_summary = str(model)

            if prompt_option and model_summary != output:
                if (
                    input(
                        f"File '{file}' does not match test output. Should this file be replaced? (y/N) "
                    ).lower()
                    == "y"
                ):
                    with open(file, "w") as f:
                        data = OrderedDict(
                            params=OrderedDict(params), output=literal(model_summary)
                        )
                        yaml.dump(data, f)
                        output = model_summary

            assert model_summary == output
