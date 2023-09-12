from .base import BaseLLM
from .openai import OpenAILLM

LLM_CLASSES = {
    cls.__name__: cls for cls in BaseLLM.__subclasses__()
}
LLM_CLASSES["default"] = OpenAILLM
