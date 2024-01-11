import os
import random

import numpy as np
import pytest
import json
import pandas as pd

from canopy.models.data_models import Document
from canopy_cli.data_loader.data_loader import (
    load_from_path,
    _load_single_schematic_file_by_suffix, _df_to_documents,
)
from canopy_cli.data_loader.errors import (
    DataLoaderException,
    DocumentsValidationError,
    IDsNotUniqueError)
from tests.unit import random_words


good_df_minimal = (
    pd.DataFrame(
        [
            {"id": 1, "text": "foo"},
            {"id": 2, "text": "bar"},
            {"id": 3, "text": "baz"},
        ]
    ),
    [
        Document(id=1, text="foo"),
        Document(id=2, text="bar"),
        Document(id=3, text="baz"),
    ]
)


good_df_all_good_metadata_permutations = (
    pd.DataFrame(
        [
            {"id": 1, "text": "foo", "metadata": {"string": "string"}},
            {"id": 2, "text": "bar", "metadata": {"int": 1}},
            {"id": 3, "text": "baz", "metadata": {"float": 1.0}},
            {"id": 4, "text": "foo", "metadata": {"list": ["list", "another"]}},
        ]
    ),
    [
        Document(id=1, text="foo", metadata={"string": "string"}),
        Document(id=2, text="bar", metadata={"int": 1}),
        Document(id=3, text="baz", metadata={"float": 1.0}),
        Document(id=4, text="foo", metadata={"list": ["list", "another"]}),
    ]
)


good_df_maximal = (
    pd.DataFrame(
        [
            {"id": 1, "text": "foo", "source": "foo_source",
             "metadata": {"foo": "foo"}},
            {"id": 2, "text": "bar", "source": "bar_source",
             "metadata": {"bar": "bar", "ndarray": np.array([1, 2, 3])}},
            {"id": 3, "text": "baz", "source": "baz_source",
             "metadata": {"baz": "baz", "list": ["item1", "item2"]}},
        ]
    ),
    [
        Document(id=1, text="foo", source="foo_source", metadata={"foo": "foo"}),
        Document(id=2, text="bar", source="bar_source",
                 metadata={"bar": "bar", "ndarray": [1, 2, 3]}),
        Document(id=3, text="baz", source="baz_source",
                 metadata={"baz": "baz", "list": ["item1", "item2"]}),
    ]
)

bad_df_missing_field = (
    pd.DataFrame(
        [
            {"id": 1, "text": "foo"},
            {"id": 2, "text": "bar"},
            {"id": 3, "text": None},
        ]
    ),
    DocumentsValidationError,
)

bad_df_bad_type = (
    pd.DataFrame(
        [
            {"id": 1, "text": {}},
            {"id": 2, "text": "bar"},
            {"id": 3, "text": 3},
        ]
    ),
    DocumentsValidationError,
)

bad_df_bad_type_optional = (
    pd.DataFrame(
        [
            {"id": 1, "text": "foo", "source": {}},
            {"id": 2, "text": "bar", "source": "bar_source"},
            {"id": 3, "text": "baz", "source": "baz_source"},
        ],
    ),
    DocumentsValidationError,
)

bad_df_bad_type_metadata = (
    pd.DataFrame(
        [
            {"id": 1, "text": "foo", "metadata": "foo"},
            {"id": 2, "text": "bar", "metadata": {"bar": "bar"}},
            {"id": 3, "text": "baz", "metadata": {"baz": "baz"}},
        ]
    ),
    DocumentsValidationError,
)

bad_df_bad_type_metadata_list = (
    pd.DataFrame(
        [
            {"id": 1, "text": "foo", "metadata": ["foo"]},
            {"id": 2, "text": "bar", "metadata": {"bar": "bar"}},
            {"id": 3, "text": "baz", "metadata": {"baz": "baz"}},
        ],
    ),
    DocumentsValidationError,
)

bad_df_metadata_not_allowed_all_permutations = (
    pd.DataFrame(
        [
            {"id": 1, "text": "foo", "metadata": {"list_of_int": [1, 2, 3]}},
            {"id": 2, "text": "bar", "metadata": {"list_of_float": [1.0, 2.0, 3.0]}},
            {"id": 3, "text": "baz", "metadata": {"dict": {"key": "value"}}},
            {"id": 4, "text": "foo", "metadata": {"list_of_dict": [{"key": "value"}]}},
            {"id": 5, "text": "bar", "metadata": {"list_of_list": [["value"]]}},
            {"id": 6, "text": "baz", "metadata": {1: "foo"}},
        ]
    ),
    DocumentsValidationError
)


bad_df_has_excess_field = (
    pd.DataFrame(
        [
            {
                "id": 1,
                "text": "foo",
                "source": "foo_source",
                "metadata": {"foo": "foo"},
                "excess": "excess",
            },
            {
                "id": 2,
                "text": "bar",
                "source": "bar_source",
                "metadata": {"bar": "bar"},
                "excess": "excess",
            },
            {
                "id": 3,
                "text": "baz",
                "source": "baz_source",
                "metadata": {"baz": "baz"},
                "excess": "excess",
            },
        ]
    ),
    DocumentsValidationError,
)

bad_df_misspelled_optional_field = (
    pd.DataFrame(
        [
            {"id": 1, "text": "foo", "sorce": "foo_source"},
            {"id": 2, "text": "bar", "metdata": {"key": "value"}},
            {"id": 3, "text": "baz", "sorce": "baz_source"},
        ]
    ),
    DocumentsValidationError
)

bad_df_missing_mandatory_field = (
    pd.DataFrame(
        [
            {"text": "foo", "metadata": {"foo": "foo"}},
            {"text": "bar", "metadata": {"bar": "bar"}},
            {"text": "baz", "metadata": {"baz": "baz"}},
        ]
    ),
    DocumentsValidationError,
)

bad_df_duplicate_ids = (
    pd.DataFrame(
        [
            {"id": 1, "text": "foo", "metadata": {"foo": "foo"}},
            {"id": 2, "text": "bar", "metadata": {"bar": "bar"}},
            {"id": 2, "text": "baz", "metadata": {"baz": "baz"}},
        ]
    ),
    IDsNotUniqueError,
)


all_dataframes_as_dict_with_name = [
    ("good_df_minimal", good_df_minimal, None),
    ("good_df_maximal", good_df_maximal, None),
    ("bad_df_missing_field", bad_df_missing_field, DataLoaderException),
    ("bad_df_bad_type", bad_df_bad_type, DataLoaderException),
    ("bad_df_bad_type_optional", bad_df_bad_type_optional, DataLoaderException),
    ("bad_df_bad_type_metadata", bad_df_bad_type_metadata, DocumentsValidationError),
    ("bad_df_bad_type_metadata_list", bad_df_bad_type_metadata_list, DocumentsValidationError),  # noqa: E501
    ("bad_df_has_excess_field", bad_df_has_excess_field, DataLoaderException),
    ("bad_df_missing_mandatory_field", bad_df_missing_mandatory_field, DocumentsValidationError),  # noqa: E501
    ("bad_df_duplicate_ids", bad_df_duplicate_ids, IDsNotUniqueError),
    ("bad_df_misspelled_optional_field", bad_df_misspelled_optional_field, DataLoaderException),  # noqa: E501
    ("good_df_all_good_metadata_permutations", good_df_all_good_metadata_permutations, None),  # noqa: E501
]


@pytest.mark.parametrize("name, df_and_expected, expected_error",
                         all_dataframes_as_dict_with_name)
def test_df_to_documents(name, df_and_expected, expected_error) -> None:
    df, expected = df_and_expected
    if name.startswith("good"):
        assert _df_to_documents(df) == expected
    else:
        with pytest.raises(expected_error):
            _df_to_documents(df)


@pytest.fixture
def dict_rows_input():
    return [
        {"id": 1, "text": "foo"},
        {"id": 2, "text": "bar", "source": "bar_source",
         "metadata": {"bar": "bar", "list": ["item1", "item2"]}},
        {"id": 3, "text": "baz",
         "metadata": {"baz": "baz", "null_val": None}},
        {"id": 4, "text": "qux", "source": "qux_source"},
    ]


@pytest.fixture
def expected_documents():
    return [
        Document(id=1, text="foo"),
        Document(id=2, text="bar", source="bar_source",
                 metadata={"bar": "bar", "list": ["item1", "item2"]}),
        Document(id=3, text="baz", metadata={"baz": "baz"}),
        Document(id=4, text="qux", source="qux_source"),
    ]


def test_load_single_file_jsonl(tmpdir, dict_rows_input, expected_documents):
    path = tmpdir.join("test.jsonl")
    path.write("\n".join([json.dumps(row) for row in dict_rows_input]))

    docs = _load_single_schematic_file_by_suffix(str(path))
    assert docs == expected_documents


def test_load_single_file_parquet(tmpdir, dict_rows_input, expected_documents):
    data = pd.DataFrame(dict_rows_input)

    path = tmpdir.join("test.parquet")
    pd.DataFrame(data).to_parquet(str(path))

    docs = _load_single_schematic_file_by_suffix(str(path))
    assert docs == expected_documents


def test_load_multiple_files(tmpdir, dict_rows_input, expected_documents):
    data2 = pd.DataFrame([
        {"id": 5, "text": "quux"},
        {"id": 6, "text": "corge", "source": "corge_source",
         "metadata": {"corge": "corge"}},
        {"id": 7, "text": "grault",
         "metadata": {"grault": "grault"}},
    ])

    expected = expected_documents + \
        [
            Document(id=5, text="quux"),
            Document(id=6, text="corge", source="corge_source",
                     metadata={"corge": "corge"}),
            Document(id=7, text="grault", metadata={"grault": "grault"}),
        ]

    tmpdir.mkdir("test_multi_files")
    base_path = tmpdir.join("test_multi_files")
    path1 = base_path.join("test1.jsonl")
    path2 = base_path.join("test2.parquet")
    path1.write("\n".join([json.dumps(row) for row in dict_rows_input]))
    pd.DataFrame(data2).to_parquet(str(path2))

    docs = load_from_path(str(base_path))
    assert sorted(docs, key=lambda x: x.id) == sorted(expected, key=lambda x: x.id)


def _generate_text(num_words: int, num_rows: int) -> str:
    return "\n".join([" ".join(random.choices(random_words, k=num_words)) for _ in range(num_rows)])  # noqa: E501


def test_load_text_files(tmpdir, dict_rows_input, expected_documents):
    tmpdir.mkdir("test_text_files")
    base_path = tmpdir.join("test_text_files")
    path1 = base_path.join("test1.jsonl")
    path1.write("\n".join([json.dumps(row) for row in dict_rows_input]))
    path2 = base_path.join("test2.txt")
    path_2_text = _generate_text(10, 3)
    path2.write(path_2_text)
    path3 = base_path.join("test3.txt")
    path_3_text = _generate_text(10, 3)
    path3.write(path_3_text)

    expected = expected_documents + [
        Document(text=path_2_text,
                 id="test2",
                 source=os.path.join(str(base_path), "test2.txt")
                 ),
        Document(text=path_3_text,
                 id="test3",
                 source=os.path.join(str(base_path), "test3.txt")
                 ),
    ]

    docs = load_from_path(str(base_path))
    assert sorted(docs, key=lambda x: x.id) == sorted(expected, key=lambda x: x.id)
