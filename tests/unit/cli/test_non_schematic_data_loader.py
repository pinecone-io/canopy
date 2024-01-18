import pytest

import pandas as pd
from pandas.testing import assert_frame_equal

from canopy_cli.data_loader.data_loader import (
    DataLoaderException,
    _load_multiple_txt_files
)


@pytest.fixture
def two_valid_txt_files(tmpdir):
    file1 = tmpdir.join("file1.txt")
    file1.write("the little brown fox\njumped over the lazy dog")
    file2 = tmpdir.join("file2.txt")
    file2.write("meow meow meow\nmeow meow meow")
    return [file1, file2]


@pytest.fixture
def invalid_txt_file(tmpdir):
    file_path = tmpdir.join("file.txt")
    with open(str(file_path), 'w', encoding='latin-1') as file:
        file.write("This is a text with bad encoding for UTF-8. ñáéíóú")
    return [file_path]


@pytest.fixture
def one_invalid_rest_is_valid(tmpdir):
    file1 = tmpdir.join("file1.txt")
    file1.write("the little brown fox\njumped over the lazy dog")
    file2 = tmpdir.join("file2.txt")
    file2.write("meow meow meow\nmeow meow meow")
    file3 = tmpdir.join("file3.txt")
    with open(str(file3), 'w', encoding='latin-1') as file:
        file.write("This is a text with bad encoding for UTF-8. ñáéíóú")
    return [file1, file2, file3]


def test_loading_files_good(two_valid_txt_files):
    expected = pd.DataFrame([
        {
            "id": "file1",
            "text": "the little brown fox\njumped over the lazy dog",
            "source": str(two_valid_txt_files[0])
        },
        {
            "id": "file2",
            "text": "meow meow meow\nmeow meow meow",
            "source": str(two_valid_txt_files[1])
        }
    ])
    docs = _load_multiple_txt_files(two_valid_txt_files)
    assert isinstance(docs, pd.DataFrame)
    assert_frame_equal(docs, expected)


def test_loading_files_bad(invalid_txt_file):
    with pytest.raises(DataLoaderException) as e:
        _load_multiple_txt_files(invalid_txt_file)
    assert str(e.value) == f"""
        {invalid_txt_file[0]}, line * - File must be UTF-8 encoded
        """


def test_loading_file_one_is_corrupted(one_invalid_rest_is_valid):
    expected = pd.DataFrame([
        {
            "id": "file1",
            "text": "the little brown fox\njumped over the lazy dog",
            "source": str(one_invalid_rest_is_valid[0])
        },
        {
            "id": "file2",
            "text": "meow meow meow\nmeow meow meow",
            "source": str(one_invalid_rest_is_valid[1])
        }
    ])
    with pytest.raises(DataLoaderException) as e:
        _load_multiple_txt_files(one_invalid_rest_is_valid)
    assert str(e.value) == f"""
        {one_invalid_rest_is_valid[2]}, line * - File must be UTF-8 encoded
        """
    remaining = _load_multiple_txt_files(one_invalid_rest_is_valid[:-1])
    assert isinstance(remaining, pd.DataFrame)
    assert_frame_equal(remaining, expected)
