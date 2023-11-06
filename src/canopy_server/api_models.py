from typing import Optional, List

from pydantic import BaseModel

from canopy.models.data_models import Messages, Query, Document


class ChatRequest(BaseModel):
    model: str = ""
    messages: Messages
    stream: bool = False
    user: Optional[str] = None

    class Config:
        extra = "ignore"


class ContextQueryRequest(BaseModel):
    queries: List[Query]
    max_tokens: int


class ContextUpsertRequest(BaseModel):
    documents: List[Document]
    batch_size: int = 200


class ContextDeleteRequest(BaseModel):
    document_ids: List[str]


class HealthStatus(BaseModel):
    pinecone_status: str
    llm_status: str


class ChatDebugInfo(BaseModel):
    id: str
    duration_in_sec: float
    intenal_model: str
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None

    def to_text(self,):
        return self.json()
