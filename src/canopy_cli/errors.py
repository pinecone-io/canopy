import click
from click import ClickException

from canopy_cli.data_loader.data_loader import format_multiline


class CLIError(ClickException):
    def format_message(self) -> str:
        return click.style(format_multiline(self.message), fg='red')


class ConfigError(RuntimeError):
    pass
