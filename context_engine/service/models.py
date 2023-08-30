from typing import Optional, List

from pydantic import BaseModel

from context_engine.models.data_models import Messages, Query


class ChatRequest(BaseModel):
    model: str = ""
    messages: Messages
    stream: bool = False
    user: Optional[str] = None


class ContextQueryRequest(BaseModel):
    queries: List[Query]
    max_tokens: Optional[int] = None
