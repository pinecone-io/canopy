import pytest
import json
import pandas as pd
from pandas.testing import assert_frame_equal
from typing import List

from resin.models.data_models import Document
from resin_cli.data_loader.data_loader import (
    IDsNotUniqueError,
    DocumentsValidationError,
    load_from_path,
    _load_single_file_by_suffix, _df_to_documents,
)


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

good_df_maximal = (
    pd.DataFrame(
        [
            {"id": 1, "text": "foo", "source": "foo_source", "metadata": {"foo": "foo"}},
            {"id": 2, "text": "bar", "source": "bar_source", "metadata": {"bar": "bar"}},
            {"id": 3, "text": "baz", "source": "baz_source", "metadata": {"baz": "baz"}},
        ]
    ),
    [
        Document(id=1, text="foo", source="foo_source", metadata={"foo": "foo"}),
        Document(id=2, text="bar", source="bar_source", metadata={"bar": "bar"}),
        Document(id=3, text="baz", source="baz_source", metadata={"baz": "baz"}),
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

bad_df_bad_type_metadata_list_int = (
    pd.DataFrame(
        [
            {"id": 1, "text": "foo", "metadata": {"foo": [1]}},
            {"id": 2, "text": "bar", "metadata": {"bar": "bar"}},
            {"id": 3, "text": "baz", "metadata": {"baz": "baz"}},
        ]
    ),
    DocumentsValidationError,
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
    ("good_df_minimal", good_df_minimal),
    ("good_df_maximal", good_df_maximal),
    ("bad_df_missing_field", bad_df_missing_field),
    ("bad_df_bad_type", bad_df_bad_type),
    ("bad_df_bad_type_optional", bad_df_bad_type_optional),
    ("bad_df_bad_type_metadata", bad_df_bad_type_metadata),
    ("bad_df_bad_type_metadata_list", bad_df_bad_type_metadata_list),
    ("bad_df_bad_type_metadata_list_int", bad_df_bad_type_metadata_list_int),
    ("bad_df_has_excess_field", bad_df_has_excess_field),
    ("bad_df_missing_mandatory_field", bad_df_missing_mandatory_field),
    ("bad_df_duplicate_ids", bad_df_duplicate_ids),
]
# Add your correct import paths and defined data frames here


@pytest.mark.parametrize("name, df_and_expected", all_dataframes_as_dict_with_name)
def test_df_to_documents(name, df_and_expected) -> None:
    df, expected = df_and_expected
    if name.startswith("good"):
        assert _df_to_documents(df) == expected
    else:
        with pytest.raises(expected):
            _df_to_documents(df)


# Other bad dataframe tests...
# Define additional test functions to cover all of your "bad" dataframes

def test_load_single_file_jsonl(tmpdir):
    data = [
        #...Your data here...
    ]

    path = tmpdir.join("test.jsonl")
    path.write("\n".join([json.dumps(row) for row in data]))

    docs = _load_single_file_by_suffix(str(path))
    assert isinstance(docs, List) and all(isinstance(doc, Document) for doc in docs)

def test_load_single_file_parquet(tmpdir):
    data = [
        #...Your data here...
    ]

    path = tmpdir.join("test.parquet")
    pd.DataFrame(data).to_parquet(str(path))

    docs = _load_single_file_by_suffix(str(path))
    assert isinstance(docs, List) and all(isinstance(doc, Document) for doc in docs)

def test_load_multiple_files_jsonl(tmpdir):
    data1 = [
        #...Your data1 here...
    ]
    data2 = [
        #...Your data2 here...
    ]

    tmpdir.mkdir("test_jsonl")
    base_path = tmpdir.join("test_jsonl")
    path1 = base_path.join("test1.jsonl")
    path2 = base_path.join("test2.jsonl")
    path1.write("\n".join([json.dumps(row) for row in data1]))
    path2.write("\n".join([json.dumps(row) for row in data2]))

    docs = load_from_path(str(base_path))
    # Your assert should now check the validity and integrity of the Document objects returned, not DataFrames.
    # Modify the assert accordingly
