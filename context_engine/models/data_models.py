from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict, Sequence

from pydantic import BaseModel, Field

Metadata = Dict[str, Union[str, int, float, List[str]]]


class Query(BaseModel):
    text: str
    namespace: str = ""
    metadata_filter: Optional[dict]
    top_k: Optional[int]
    query_params: Optional[dict] = Field(default_factory=dict)


class Document(BaseModel):
    id: str
    text: str
    metadata: Metadata


class ContextContent(BaseModel, ABC):

    # Any context should be able to be represented as well formatted text.
    # In the most minimal case, that could simply be a call to `.json()`.
    @abstractmethod
    def to_text(self) -> str:
        pass


class Context(BaseModel):
    content: Union[ContextContent, Sequence[ContextContent]]
    num_tokens: int = Field(exclude=True)
    debug_info: dict = Field(default_factory=dict, exclude=True)

    def to_text(self) -> str:
        if isinstance(self.content, ContextContent):
            return self.content.to_text()
        else:
            return "\n".join([c.to_text() for c in self.content])

# TODO: add ChatEngine main models - `Messages`, `Answer`
