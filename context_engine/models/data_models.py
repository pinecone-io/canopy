from typing import Optional, List

from pydantic import BaseModel, Field


class Query(BaseModel):
    text: str
    namespace: str = ""
    metadata_filter: Optional[dict]
    top_k: int


class ContextDocument(BaseModel):
    reference: str
    text: str


class ContextQueryResult(BaseModel):
    query: str
    documents: List[ContextDocument]


class Context(BaseModel):
    results: List[ContextQueryResult]
    num_tokens: int = Field(exclude=True)
    debug_info: dict = Field(default_factory=dict, exclude=True)

    def json(self, *args, **kwargs):
        # TODO: consider formatting as pure text, without JSON syntax
        return super().json(*args, **kwargs)
