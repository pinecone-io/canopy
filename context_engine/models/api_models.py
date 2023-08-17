from datetime import datetime
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from context_engine.llm.models import AssistantMessage
from context_engine.models.data_models import MessageBase, LLMResponse


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

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    model: str
    choices: Sequence[_Choice]
    usage: TokenCounts

    @classmethod
    def from_llm_response(cls,
                          llm_response: LLMResponse,
                          model: str,
                          ):
        return cls(id=llm_response.id,
                   model=model,
                   choices=[
                       _Choice(message=AssistantMessage(content=msg, index=i)
                       for i, msg in enumerate(llm_response.choices))
                   ],
                   usage=TokenCounts(prompt_tokens=llm_response.prompt_tokens,
                                      completion_tokens=llm_response.completion_tokens),
                   )


class StreamingChatResponse(BaseModel):
    id: str
    object: str = "chat.chunk"
    created: Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    model: str
