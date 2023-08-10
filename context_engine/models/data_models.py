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


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Metadata
    values: Optional[List[float]] = None
    sparse_values: Optional[Dict[str, Union[int, float]]] = None


class DocumentChunkWithScore(DocumentChunk):
    score: float


class QueryResult(BaseModel):
    query: str
    documents: List[Document]


class ContextSnippet(BaseModel):
    reference: str
    text: str


class ContextQueryResult(BaseModel):
    query: str
    snippets: List[ContextSnippet]


class Context(BaseModel):
    result: Union[str, BaseModel, Iterable[BaseModel]]
    num_tokens: int = Field(exclude=True)
    debug_info: dict = Field(default_factory=dict, exclude=True)

    def json(self, *args, **kwargs):
        # TODO: consider formatting as pure text, without JSON syntax
        return super().json(*args, **kwargs)


