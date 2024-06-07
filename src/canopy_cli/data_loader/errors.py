import typing as t
from click._compat import get_text_stderr
from click import echo


class IDsNotUniqueError(ValueError):
    pass


class DocumentsValidationError(ValueError):
    pass


class DataLoaderException(Exception):
    """An exception that Click can handle and show to the user."""

    #: The exit code for this exception.
    exit_code = 1

    def __init__(self, file_name: str, row_id: str, err: str) -> None:
        message = f"""
        {file_name}, line {row_id} - {err}
        """
        super().__init__(message)
        self.file_name = file_name
        self.row_id = row_id
        self.err = err

    def format_message(self) -> str:
        message = f"""
        {self.file_name}, line {self.row_id} - {self.err}
        """
        return message

    def __str__(self) -> str:
        return self.format_message()

    def show(self, file: t.Optional[t.IO] = None) -> None:
        if file is None:
            file = get_text_stderr()

        echo("{message}".format(message=self.format_message()), file=file)
