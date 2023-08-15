from typing import Optional, List, Union, Iterable, Dict

from pydantic import BaseModel, Field

Metadata = Dict[str, Union[str, int, float, List[str]]]


class Query(BaseModel):
    text: str
    namespace: str = ""
    metadata_filter: Optional[dict]
    top_k: int


class Document(BaseModel):
    id: str
    text: str
    metadata: Metadata


class Context(BaseModel):
    content: Union[str, BaseModel, Iterable[BaseModel]]
    num_tokens: int = Field(exclude=True)
    debug_info: dict = Field(default_factory=dict, exclude=True)

# TODO: add ChatEngine main models - `Messages`, `Answer`
