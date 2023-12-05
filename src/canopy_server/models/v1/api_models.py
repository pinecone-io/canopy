from typing import Optional, List

from pydantic import BaseModel, Field

from canopy.models.data_models import Messages, Query, Document

# TODO: consider separating these into modules: Chat, Context, Application, etc.


class ChatRequest(BaseModel):
    model: str = Field(
        default="",
        description="The ID of the model to use. This field is ignored; instead, configure this field in the Canopy config.",  # noqa: E501
    )
    messages: Messages = Field(
        description="A list of messages comprising the conversation so far."
    )
    stream: bool = Field(
        default=False,
        description="""Whether or not to stream the chatbot's response. If set, the response is server-sent events containing [chat.completion.chunk](https://platform.openai.com/docs/api-reference/chat/streaming) objects""",  # noqa: E501
    )
    user: Optional[str] = Field(
        default=None,
        description="A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Unused, reserved for future extensions",  # noqa: E501
    )

    class Config:
        extra = "ignore"


class ContextQueryRequest(BaseModel):
    queries: List[Query]
    max_tokens: int


class ContextResponse(BaseModel):
    content: str
    num_tokens: int


class ContextUpsertRequest(BaseModel):
    documents: List[Document]
    batch_size: int = Field(
        default=200, description="The batch size to use when uploading documents chunks to the Pinecone Index."  # noqa: E501
    )


class ContextDeleteRequest(BaseModel):
    document_ids: List[str] = Field(description="List of document IDs to delete.")


class HealthStatus(BaseModel):
    pinecone_status: str
    llm_status: str


class ChatDebugInfo(BaseModel):
    id: str
    duration_in_sec: float
    internal_model: str
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
