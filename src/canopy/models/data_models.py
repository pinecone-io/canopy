from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Union, Dict, Sequence

from pydantic import BaseModel, Field, validator, Extra

Metadata = Dict[str, Union[str, int, float, List[str]]]


# ----------------- Context Engine models -----------------


class Query(BaseModel):
    text: str
    namespace: str = ""
    metadata_filter: Optional[dict] = None
    top_k: Optional[int] = None
    query_params: dict = Field(default_factory=dict)


class Document(BaseModel):
    id: str
    text: str
    source: str = ""
    metadata: Metadata = Field(default_factory=dict)

    class Config:
        extra = Extra.forbid

    @validator('metadata')
    def metadata_reseved_fields(cls, v):
        if 'text' in v:
            raise ValueError('Metadata cannot contain reserved field "text"')
        if 'document_id' in v:
            raise ValueError('Metadata cannot contain reserved field "document_id"')
        if 'source' in v:
            raise ValueError('Metadata cannot contain reserved field "source"')
        return v


class ContextContent(BaseModel, ABC):

    # Any context should be able to be represented as well formatted text.
    # In the most minimal case, that could simply be a call to `.json()`.
    @abstractmethod
    def to_text(self, **kwargs) -> str:
        pass


class Context(BaseModel):
    content: Union[ContextContent, Sequence[ContextContent]]
    num_tokens: int = Field(exclude=True)
    debug_info: dict = Field(default_factory=dict, exclude=True)

    def to_text(self, **kwargs) -> str:
        if isinstance(self.content, ContextContent):
            return self.content.to_text(**kwargs)
        else:
            return "\n".join([c.to_text(**kwargs) for c in self.content])


# --------------------- LLM models ------------------------


class Role(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class MessageBase(BaseModel):
    role: Role
    content: str

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        d['role'] = d['role'].value
        return d


Messages = List[MessageBase]
