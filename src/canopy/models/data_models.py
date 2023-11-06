from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Union, Dict, Literal

from pydantic import BaseModel, Field, validator, Extra

Metadata = Dict[str, Union[str, int, float, List[str]]]


# ----------------- Context Engine models -----------------


class Query(BaseModel):
    text: str = Field(description="The query text.")
    namespace: str = Field(
        default="",
        description="The namespace of the query. To learn more about namespaces, see https://docs.pinecone.io/docs/namespaces",  # noqa: E501
    )
    metadata_filter: Optional[dict] = Field(
        default=None,
        description="A Pinecone metadata filter, to learn more about metadata filters, see https://docs.pinecone.io/docs/metadata-filtering",  # noqa: E501
    )
    top_k: Optional[int] = Field(
        default=None,
        description="The number of results to return."
    )
    query_params: dict = Field(
        default_factory=dict,
        description="Pinecone Client additional query parameters."
    )


class Document(BaseModel):
    id: str = Field(description="The document id.")
    text: str = Field(description="The document text.")
    source: str = Field(
        default="",
        description="The source of the document: a URL, a file path, etc."
    )
    metadata: Metadata = Field(
        default_factory=dict,
        description="The document metadata. To learn more about metadata, see https://docs.pinecone.io/docs/manage-data",  # noqa: E501
    )

    class Config:
        extra = Extra.forbid

    @validator("metadata")
    def metadata_reseved_fields(cls, v):
        if "text" in v:
            raise ValueError('Metadata cannot contain reserved field "text"')
        if "document_id" in v:
            raise ValueError('Metadata cannot contain reserved field "document_id"')
        if "source" in v:
            raise ValueError('Metadata cannot contain reserved field "source"')
        return v


class ContextContent(BaseModel, ABC):
    # Any context should be able to be represented as well formatted text.
    # In the most minimal case, that could simply be a call to `.json()`.
    @abstractmethod
    def to_text(self, **kwargs) -> str:
        pass

    def __str__(self):
        return self.to_text()


class Context(BaseModel):
    content: ContextContent
    num_tokens: int
    debug_info: dict = Field(default_factory=dict, exclude=True)

    def to_text(self, **kwargs) -> str:
        return self.content.to_text(**kwargs)


# --------------------- LLM models ------------------------


class Role(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class MessageBase(BaseModel):
    role: Role = Field(description="The role of the message's author. "
                                   "Can be one of ['User', 'Assistant', 'System']")
    content: str = Field(description="The contents of the message.")

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        d["role"] = d["role"].value
        return d


Messages = List[MessageBase]


class UserMessage(MessageBase):
    role: Literal[Role.USER] = Role.USER
    content: str


class SystemMessage(MessageBase):
    role: Literal[Role.SYSTEM] = Role.SYSTEM
    content: str


class AssistantMessage(MessageBase):
    role: Literal[Role.ASSISTANT] = Role.ASSISTANT
    content: str
