from pathlib import Path
import typer

from cookiecutter.main import cookiecutter as cookiecutter_main

app = typer.Typer()


@app.command()
def cookiecutter(
    template: str = "",
    checkout: str = None,
    no_input: bool = False,
    extra_context: str = None,
    replay: bool = False,
    overwrite_if_exists: bool = False,
    output_dir: Path = ".",
    config_file: Path = None,
    default_config: bool = False,
    directory: str = None,
    skip_if_file_exists: bool = False,
):
    """
    Generate a fastapp project using cookiecutter.
    """
    if not template:
        template = Path(__file__).parent.resolve() / "cookiecutter"
    elif template in ["gh", "github"]:
        template = "https://github.com/rbturnbull/fastapp-cookiecutter"

    return cookiecutter_main(
        template=str(template),
        checkout=checkout,
        no_input=no_input,
        extra_context=extra_context,
        replay=replay,
        overwrite_if_exists=overwrite_if_exists,
        output_dir=output_dir,
        config_file=config_file,
        default_config=default_config,
        directory=directory,
        skip_if_file_exists=skip_if_file_exists,
    )
