from datetime import datetime
from typing import Optional, Sequence

from pydantic import BaseModel, Field, validator

from context_engine.models.data_models import MessageBase


class _Choice(BaseModel):
    index: int = 0
    message: MessageBase
    finish_reason: str = "stop"


class _StreamChoice(BaseModel):
    index: int = 0
    delta: MessageBase
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
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    model: str
    choices: Sequence[_Choice]
    usage: TokenCounts
    debug_info: dict = Field(default_factory=dict, exclude=True)


class StreamingChatResponse(BaseModel):
    id: str
    object: str = "chat.chunk"
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    model: str
