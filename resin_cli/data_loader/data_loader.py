import os
import glob
import pandas as pd

from pydantic import ValidationError

from resin.models.data_models import Document


class IndexNotUniqueError(ValueError):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


def _validate_dataframe(df: pd.DataFrame) -> bool:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dataframe must be a pandas DataFrame")
    if not df.index.is_unique:
        raise IndexNotUniqueError("Dataframe index must be unique")
    for row in df.to_dict(orient="records"):
        try:
            Document.validate(row)

        # if any row fails validation, return False
        except ValidationError as e:
            return False
        except ValueError as e:
            raise e
    # all rows validated
    return True


def _load_single_file_by_suffix(f: str) -> pd.DataFrame:
    if f.endswith(".parquet"):
        df = pd.read_parquet(f)
    elif f.endswith(".jsonl"):
        df = pd.read_json(f, lines=True)
    else:
        raise ValueError("Only .parquet and .jsonl files are supported")

    return df


def load_dataframe_from_path(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        all_files = glob.glob(os.path.join(path, "*.jsonl")) + glob.glob(
            os.path.join(path, "*.parquet")
        )
        df = pd.concat(
            [_load_single_file_by_suffix(f) for f in all_files],
            axis=0,
            ignore_index=True,
        )
    else:
        df = _load_single_file_by_suffix(path)

    if not _validate_dataframe(df):
        raise ValueError("Dataframe failed validation")

    return df
