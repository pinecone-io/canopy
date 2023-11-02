from typing import Optional, List

from pydantic import BaseModel, Field

from canopy.models.data_models import Messages, Query, Document


class ChatRequest(BaseModel):
    model: str = Field(
        default="",
        description="ID of the model to use. Currecntly this field is ignored and this should be configured on Canopy config.",  # noqa: E501
    )
    messages: Messages = Field(
        description="A list of messages comprising the conversation so far."
    )
    stream: bool = Field(
        default=False,
        description="""If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message.""",  # noqa: E501
    )
    user: Optional[str] = Field(
        default=None,
        description="A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.",  # noqa: E501
    )


class ContextQueryRequest(BaseModel):
    queries: List[Query]
    max_tokens: int


class ContextUpsertRequest(BaseModel):
    documents: List[Document]
    batch_size: int = Field(
        default=200, description="Batch size for upserting documents to Pinecone."
    )


class ContextDeleteRequest(BaseModel):
    document_ids: List[str] = Field(description="List of document ids to delete.")


class HealthStatus(BaseModel):
    pinecone_status: str
    llm_status: str


class ChatDebugInfo(BaseModel):
    id: str
    duration_in_sec: float
    intenal_model: str
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None

    def to_text(
        self,
    ):
        return self.json()


class ShutdownResponse(BaseModel):
    message: str = Field(
        default="Shutting down",
        description="Message indicating the server is shutting down.",
    )


class SuccessUpsertResponse(BaseModel):
    message: str = Field(
        default="Success",
        description="Message indicating the upsert was successful.",
    )


class SuccessDeleteResponse(BaseModel):
    message: str = Field(
        default="Success",
        description="Message indicating the delete was successful.",
    )
