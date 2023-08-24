from typing import Literal, Optional, List, Union

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


class ModelParams(BaseModel):
    temperature: Optional[float]
    n: Optional[int]  # number of completions to generate
    top_p: Optional[float]


class FunctionPrimitiveProperty(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None


class FunctionArrayProperty(BaseModel):
    name: str
    items_type: str
    # we require description for array properties
    # because the model is more struggling with them
    description: str

    def dict(self, *args, **kwargs):
        super_dict = super().dict(*args, **kwargs)
        if "items_type" in super_dict:
            super_dict["items"] = {"type": super_dict.pop("items_type")}
        return super_dict


FunctionProperty = Union[FunctionPrimitiveProperty, FunctionArrayProperty]


class FunctionParameters(BaseModel):
    required_properties: List[FunctionProperty]
    optional_properties: List[FunctionProperty] = []

    def dict(self, *args, **kwargs):
        return {
            "type": "object",
            "properties": {
                pro.name: pro.dict(exclude_none=True, exclude={"name"})
                for pro in self.required_properties + self.optional_properties
            },
            "required": [pro.name for pro in self.required_properties],
        }


class Function(BaseModel):
    name: str
    description: str
    parameters: FunctionParameters
