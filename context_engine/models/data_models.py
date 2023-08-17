from enum import Enum
from typing import Optional, List, Union, Iterable, Dict, Sequence

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


class Context(BaseModel):
    content: Union[str, BaseModel, Iterable[BaseModel]]
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
