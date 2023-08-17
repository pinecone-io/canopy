from abc import ABC, abstractmethod
from ctypes import Union
from typing import Iterable

from context_engine.context_engine import ContextEngine
from context_engine.llm.base import LLM
from context_engine.models.api_models import StreamingChatResponse, ChatResponse
from context_engine.models.data_models import Context


class ChatEngine(ABC):
    def __init__(self,
                 llm: LLM,
                 context_engine: ContextEngine,
                 *,
                 max_prompt_tokens: int,
                 max_generated_tokens: int,
                 ):
        self.llm = llm
        self.context_engine = context_engine
        self.max_prompt_tokens = max_prompt_tokens
        self.max_generated_tokens = max_generated_tokens

    @abstractmethod
    def chat(self,
             messages: History,
             *,
             stream: bool = False,
             ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        pass

    @abstractmethod
    def get_context(self,
                    messages: History,
                    *,
                    stream: bool = False,
                    ) -> Context:
        pass
