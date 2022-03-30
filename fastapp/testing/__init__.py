import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--prompt",
        action="store_true",
        help="Whether or not to prompt for saving output in new expected files.",
    )


@pytest.fixture
def prompt_option(request):
    return request.config.getoption("--prompt")
