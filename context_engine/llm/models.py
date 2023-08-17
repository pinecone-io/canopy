from typing import Literal

from pydantic import BaseModel

from context_engine.models.data_models import MessageBase, Role


class UserMessage(MessageBase):
    role: Literal[Role.USER] = Role.USER
    content: str


class SystemMessage(MessageBase):
    role: Literal[Role.SYSTEM] = Role.SYSTEM
    content: str


class AssistantMessage(MessageBase):
    role: Literal[Role.ASSISTANT] = Role.ASSISTANT
    content: str


class Function(BaseModel):
    name: str
    description: str
    # TODO: decide if  we want the full FunctionParameters implementation from
    #  context-engine-exploration
    parameters: dict
