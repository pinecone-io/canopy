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

