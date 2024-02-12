from typing import Optional, Sequence, Iterable

from pydantic import BaseModel, Field, validator, ConfigDict, computed_field

from canopy.models.data_models import MessageBase


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

    @computed_field
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatResponse(BaseModel):
    id: str = Field(description="Canopy session Id.")
    object: str
    created: int
    model: str
    choices: Sequence[_Choice]
    usage: TokenCounts
    debug_info: dict = Field(default_factory=dict, exclude=True)


class StreamingChatChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: Sequence[_StreamChoice]


class StreamingChatResponse(BaseModel):
    chunks: Iterable[StreamingChatChunk]
    debug_info: dict = Field(default_factory=dict, exclude=True)
