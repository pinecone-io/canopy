from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict, Sequence, Iterable

from pydantic import BaseModel, Field

Metadata = Dict[str, Union[str, int, float, List[str]]]


# ----------------- Context Engine models -----------------


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


# TODO: add ChatEngine main models - `Messages`, `Answer`


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


class LLMResponse(BaseModel):
    id: str
    choices: Sequence[str]
    generated_tokens: Optional[int] = Field(default=None, exclude=True)
    prompt_tokens: Optional[int] = Field(default=None, exclude=True)
