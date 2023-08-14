from typing import Optional, List, Union, Iterable, Dict

from pydantic import BaseModel, Field

Metadata = Dict[str, Union[str, int, float, List[str]]]

Filter = Dict[str, Dict[str, Union[str, int, float, List[str]]]]



class Query(BaseModel):
    text: str
    namespace: str = ""
    metadata_filter: Optional[dict]
    top_k: Optional[int]
    query_params: Optional[dict]


class Document(BaseModel):
    id: str
    text: str
    metadata: Metadata

class PineconeQueryRecord(BaseModel):
    vector: Optional[List[float]] = None
    sparse_vector: Optional[List[int]] = None
    filter: Filter
    top_k: Optional[int] = 1

class PineconeDocumentRecord(BaseModel):
    id: str
    values: Optional[List[float]] = None
    sparse_values: Optional[List[int]] = None
    metadata: Metadata

class Context(BaseModel):
    content: Union[str, BaseModel, Iterable[BaseModel]]
    num_tokens: int = Field(exclude=True)
    debug_info: dict = Field(default_factory=dict, exclude=True)

    def json(self, *args, **kwargs):
        # TODO: consider formatting as pure text, without JSON syntax
        return super().json(*args, **kwargs)

# TODO: add ChatEngine main models - `Messages`, `Answer`
