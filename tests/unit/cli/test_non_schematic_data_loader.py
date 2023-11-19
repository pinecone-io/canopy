import pytest

from canopy.models.data_models import Document
from canopy_cli.data_loader.data_loader import (
    IDsNotUniqueError,
    DocumentsValidationError,
    load_from_path,
    _load_single_schematic_file_by_suffix,
    _df_to_documents,
)


@pytest.fixture
def two_valid_txt_files(tmpdir):
    file1 = tmpdir.join("file1.txt")
    file1.write("the little brown fox\njumped over the lazy dog")
    file2 = tmpdir.join("file2.txt")
    file2.write("meow meow meow\nmeow meow meow")
    return [file1, file2]


# @pytest.fixture
# def invalid_txt_file