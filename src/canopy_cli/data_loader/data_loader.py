import json
import os
import glob
from enum import Enum
from collections.abc import Iterable
from typing import List
from textwrap import dedent

import numpy as np
import pandas as pd

from pydantic import ValidationError

from canopy.models.data_models import Document
from canopy_cli.data_loader.errors import (
    DataLoaderException,
    DocumentsValidationError,
    IDsNotUniqueError)


class NonSchematicFilesTypes(Enum):
    TEXT = "txt"


def format_multiline(msg):
    return dedent(msg).strip()


def _process_metadata(value):
    if pd.isna(value):
        return {}

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise DocumentsValidationError(
                f"Metadata must be a valid json string. Error: {e}"
            ) from e

    if not isinstance(value, dict):
        raise DocumentsValidationError("Metadata must be a dict or json string")

    return {k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in value.items()
            if isinstance(v, Iterable) or pd.notna(v)}


def _df_to_documents(df: pd.DataFrame, origin_file_path=None) -> List[Document]:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dataframe must be a pandas DataFrame")
    if "id" not in df.columns:
        raise DocumentsValidationError("Missing 'id' column")
    if df.id.nunique() != df.shape[0]:
        raise IDsNotUniqueError("IDs must be unique")

    try:
        if "metadata" in df.columns:
            df.loc[:, "metadata"] = df["metadata"].apply(_process_metadata)
        documents = []
        for row in df.itertuples(index=False):
            try:
                documents.append(
                    Document(
                        **{k: v for k, v in row._asdict().items() if not pd.isna(v)}
                    )
                )
            except ValidationError as e:
                raise DataLoaderException(
                    file_name=origin_file_path,
                    row_id=row.id,
                    err=format_multiline(e.errors()[0]["msg"])
                ) from e
    except ValidationError as e:
        raise DocumentsValidationError("Documents failed validation") from e
    except ValueError as e:
        raise DocumentsValidationError(f"Unexpected error in validation: {e}") from e
    return documents


def _load_multiple_txt_files(file_paths: List[str]) -> pd.DataFrame:
    """Load multiple text files into a single dataframe

    Args:
        file_paths: List of file paths to load

    Returns:
        pd.DataFrame: Dataframe with columns `id`, `text` and 'source`
                      Note: metadata will be empty
    """
    if not isinstance(file_paths, list):
        raise ValueError("file_paths must be a list of strings")
    if len(file_paths) == 0:
        raise ValueError("file_paths must not be empty")

    rows = []
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                text = f.read()
                rows.append(
                    {
                        "id": os.path.basename(file_path).replace(".txt", ""),
                        "text": text,
                        "source": file_path
                    }
                )
        except UnicodeDecodeError as e:
            raise DataLoaderException(
                file_name=file_path,
                row_id="*",
                err="File must be UTF-8 encoded"
            ) from e
    df = pd.DataFrame(rows, columns=["id", "text", "source"])
    return df


def _load_single_schematic_file_by_suffix(file_path: str) -> List[Document]:
    try:
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".jsonl"):
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(
                "Only [.parquet, .jsonl, .csv, .txt] files are supported"
            )
    except ValueError as e:
        raise e
    except Exception as e:
        raise DataLoaderException(
            file_name=file_path,
            row_id="*",
            err=str(e)
        ) from e
    return _df_to_documents(df, origin_file_path=file_path)


def _load_multiple_non_schematic_files(
    file_paths: List[str],
    type: NonSchematicFilesTypes
) -> List[Document]:
    if not isinstance(file_paths, list):
        raise ValueError("file_paths must be a list of strings")
    if len(file_paths) == 0:
        raise ValueError("file_paths must not be empty")

    if type == NonSchematicFilesTypes.TEXT:
        df = _load_multiple_txt_files(file_paths)
    else:
        raise ValueError(f"Unsupported file type: {type}")

    return _df_to_documents(df)


def load_from_path(path: str) -> List[Document]:
    """
    Load documents from a file or directory

    Args:
        path: Path to file or directory

    Returns:
        List[Document]: List of documents
    """
    if os.path.isdir(path):
        # List all files in directory
        all_files_schematic = []
        all_files_non_schematic_txt = []
        for file in glob.glob(os.path.join(path, "*")):
            if not os.path.isfile(file):
                continue
            if file.endswith(".txt"):
                all_files_non_schematic_txt.append(file)
            elif (file.endswith(".jsonl") or
                  file.endswith(".csv") or
                  file.endswith(".parquet")):
                all_files_schematic.append(file)
        if len(all_files_schematic) + len(all_files_non_schematic_txt) == 0:
            raise ValueError("No files found in directory")

        documents: List[Document] = []
        # Load all schematic files
        for f in all_files_schematic:
            documents.extend(_load_single_schematic_file_by_suffix(f))

        # Load all non-schematic files
        if len(all_files_non_schematic_txt) > 0:
            documents.extend(
                _load_multiple_non_schematic_files(
                    all_files_non_schematic_txt,
                    NonSchematicFilesTypes.TEXT))

    # Load single file
    elif os.path.isfile(path):
        if path.endswith(".txt"):
            documents = _load_multiple_non_schematic_files(
                [path],
                NonSchematicFilesTypes.TEXT)
        else:
            documents = _load_single_schematic_file_by_suffix(path)
    else:
        raise ValueError(f"Could not find file or directory at {path}")
    return documents
