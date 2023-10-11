import json
import os
import glob
from typing import List

import pandas as pd

from pydantic import ValidationError

from resin.models.data_models import Document


class IDsNotUniqueError(ValueError):
    def __init__(self, message):
        super().__init__(message)


class DocumentsValidationError(ValueError):
    def __init__(self, message):
        super().__init__(message)


def _df_to_documents(df: pd.DataFrame) -> List[Document]:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dataframe must be a pandas DataFrame")
    if "id" not in df.columns:
        raise DocumentsValidationError("Missing 'id' column")
    if df.id.nunique() != df.shape[0]:
        raise IDsNotUniqueError("IDs must be unique")

    try:
        documents: List[Document] = []
        for row in df.to_dict(orient="records"):
            if "metadata" in row:
                if pd.isna(row["metadata"]):
                    row["metadata"] = {}
                elif type(row["metadata"]) is str:
                    try:
                        row["metadata"] = json.loads(row["metadata"])
                    except json.JSONDecodeError as e:
                        raise DocumentsValidationError(
                            f"Metadata must be a valid json string. Error: {e}"
                        ) from e
                elif type(row["metadata"]) is dict:
                    raise DocumentsValidationError(
                        "Metadata must be a dict or json string"
                    )
                row["metadata"] = {k: v for k, v in row["metadata"].items()
                                   if not pd.isna(v)}
            row = {k: v for k, v in row.items() if not pd.isna(v)}
            documents.append(Document(**row))  # type: ignore
    except ValidationError as e:
        raise DocumentsValidationError("Documents failed validation") from e
    except ValueError as e:
        raise DocumentsValidationError(f"Unexpected error in validation: {e}") from e
    return documents


def _load_single_file_by_suffix(file_path: str) -> List[Document]:
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".jsonl"):
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError("Only .parquet and .jsonl files are supported")

    return _df_to_documents(df)


def load_from_path(path: str) -> List[Document]:
    if os.path.isdir(path):
        all_files = [f for ext in ['*.jsonl', '*.parquet', '*.csv']
                     for f in glob.glob(os.path.join(path, ext))]
        if len(all_files) == 0:
            raise ValueError("No files found in directory")
        documents: List[Document] = []
        for f in all_files:
            documents.extend(_load_single_file_by_suffix(f))
    elif os.path.isfile(path):
        documents = _load_single_file_by_suffix(path)
    else:
        raise ValueError(f"Could not find file or directory at {path}")
    return documents
