from typing import Optional, Sequence

from pydantic import BaseModel, Field, validator

from context_engine.models.data_models import MessageBase


class _Choice(BaseModel):
    index: int
    message: MessageBase
    finish_reason: Optional[str] = None


class _StreamChoice(BaseModel):
    index: int
    delta: dict
    finish_reason: Optional[str] = None


class TokenCounts(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: Optional[int] = None

    @validator("total_tokens", always=True)
    def calc_total_tokens(cls, v, values, **kwargs):
        return values["prompt_tokens"] + values["completion_tokens"]


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: Sequence[_Choice]
    usage: TokenCounts
    debug_info: dict = Field(default_factory=dict, exclude=True)


class StreamingChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: Sequence[_StreamChoice]
