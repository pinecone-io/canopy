import pytest
import json
import pandas as pd
from pandas.testing import assert_frame_equal

from resin_cli.data_loader.data_loader import _validate_dataframe, IndexNotUniqueError, _load_single_file_by_suffix, load_dataframe_from_path

good_df_minimal = pd.DataFrame([
    {"id": 1, "text": "foo"},
    {"id": 2, "text": "bar"},
    {"id": 2, "text": "baz"},
])

good_df_maximal = pd.DataFrame([
    {"id": 1, "text": "foo", "source": "foo_source", "metadata": {"foo": "foo"}},
    {"id": 2, "text": "bar", "source": "bar_source", "metadata": {"bar": "bar"}},
    {"id": 3, "text": "baz", "source": "baz_source", "metadata": {"baz": "baz"}},
])

bad_df_missing_field = pd.DataFrame([
    {"id": 1, "text": "foo"},
    {"id": 2, "text": "bar"},
    {"id": 3, "text": None},
])

bad_df_bad_type = pd.DataFrame([
    {"id": 1, "text": {}},
    {"id": 2, "text": "bar"},
    {"id": 3, "text": 3},
])

bad_df_bad_type_optional = pd.DataFrame([
    {"id": 1, "text": "foo", "source": {}},
    {"id": 2, "text": "bar", "source": "bar_source"},
    {"id": 3, "text": "baz", "source": "baz_source"},
])

bad_df_bad_type_metadata = pd.DataFrame([
    {"id": 1, "text": "foo", "metadata": "foo"},
    {"id": 2, "text": "bar", "metadata": {"bar": "bar"}},
    {"id": 2, "text": "baz", "metadata": {"baz": "baz"}},
    {"id": 3, "text": "baz", "metadata": {"baz": "baz"}},
])

bad_df_bad_type_metadata_list = pd.DataFrame([
    {"id": 1, "text": "foo", "metadata": ["foo"]},
    {"id": 2, "text": "bar", "metadata": {"bar": "bar"}},
    {"id": 2, "text": "baz", "metadata": {"baz": "baz"}},
    {"id": 3, "text": "baz", "metadata": {"baz": "baz"}},
])

all_dataframes_as_dict_with_name = {
    "good_df_minimal": good_df_minimal,
    "good_df_maximal": good_df_maximal,
    "bad_df_missing_field": bad_df_missing_field,
    "bad_df_bad_type": bad_df_bad_type,
    "bad_df_bad_type_optional": bad_df_bad_type_optional,
    "bad_df_bad_type_metadata": bad_df_bad_type_metadata,
    "bad_df_bad_type_metadata_list": bad_df_bad_type_metadata_list,
}

def test_except_not_dataframe():
    """Test that _validate_dataframe raises a ValueError if not passed a dataframe."""
    with pytest.raises(ValueError):
        _validate_dataframe([
            {"id": 1, "text": "foo"},
            {"id": 2, "text": "bar"},
            {"id": 2, "text": "baz"},
        ])

def test_except_not_unique():
    """Test that _validate_dataframe raises a ValueError if passed a dataframe with a non-unique index."""
    with pytest.raises(IndexNotUniqueError):
        _validate_dataframe(pd.DataFrame([
            {"id": 1, "text": "foo"},
            {"id": 2, "text": "bar"},
            {"id": 2, "text": "baz"},
        ], index=[1, 2, 2]))

def test_all_validator_cases():
    """Test that _validate_dataframe returns True for all dataframes in all_dataframes."""
    for name, df in all_dataframes_as_dict_with_name.items():
        print(name)
        if name.startswith("bad"):
            assert _validate_dataframe(df) == False
        elif name.startswith("good"):
            assert _validate_dataframe(df) == True
        print("ok")


def test_load_single_file_jsonl(tmpdir):
    data = [
        {"id": 1, "text": "foo", "source": "foo_source", "metadata": {"foo": "foo"}},
        {"id": 2, "text": "bar", "source": "bar_source", "metadata": {"bar": "bar"}},
        {"id": 3, "text": "baz", "source": "baz_source", "metadata": {"baz": "baz"}},
    ]

    path = tmpdir.join("test.jsonl")
    path.write("\n".join([json.dumps(row) for row in data]))

    df = _load_single_file_by_suffix(str(path))
    assert_frame_equal(df, pd.DataFrame(data)) 
    

def test_load_single_file_parquet(tmpdir):
    data = [
        {"id": 1, "text": "foo", "source": "foo_source", "metadata": {"foo": "foo"}},
        {"id": 2, "text": "bar", "source": "bar_source", "metadata": {"foo": "bar"}},
        {"id": 3, "text": "baz", "source": "baz_source", "metadata": {"foo": "baz"}},
    ]

    path = tmpdir.join("test.parquet")
    pd.DataFrame(data).to_parquet(str(path))

    df = _load_single_file_by_suffix(str(path))
    assert_frame_equal(df, pd.DataFrame(data))


def test_load_multiple_files_jsonl(tmpdir):
    data1 = [
        {"id": 1, "text": "foo", "source": "foo_source", "metadata": {"foo": "foo"}},
        {"id": 2, "text": "bar", "source": "bar_source", "metadata": {"foo": "bar"}},
        {"id": 3, "text": "baz", "source": "baz_source", "metadata": {"foo": "baz"}},
    ]

    data2 = [
        {"id": 4, "text": "foofoo", "source": "foo_source", "metadata": {"foo": "foo"}},
        {"id": 5, "text": "barbar", "source": "bar_source", "metadata": {"foo": "bar"}},
        {"id": 6, "text": "bazbaz", "source": "baz_source", "metadata": {"foo": "baz"}},
    ]
    
    tmpdir.mkdir("test_jsonl")
    base_path = tmpdir.join("test_jsonl")
    path1 = base_path.join("test1.jsonl")
    path2 = base_path.join("test2.jsonl")
    path1.write("\n".join([json.dumps(row) for row in data1]))
    path2.write("\n".join([json.dumps(row) for row in data2]))

    df = load_dataframe_from_path(str(base_path))
    assert_frame_equal(df, pd.concat([pd.DataFrame(data2), pd.DataFrame(data1)], ignore_index=True))
