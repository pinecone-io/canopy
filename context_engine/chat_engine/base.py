from abc import ABC, abstractmethod
from typing import Iterable, Union

from context_engine.chat_engine.query_builder.base import QueryBuilder
from context_engine.context_engine import ContextEngine
from context_engine.llm.base import LLM
from context_engine.models.api_models import StreamingChatResponse, ChatResponse
from context_engine.models.data_models import Context, Messages


class BaseChatEngine(ABC):
    @abstractmethod
    def chat(self,
             messages: Messages,
             *,
             stream: bool = False,
             ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        pass

    @abstractmethod
    def get_context(self,
                    messages: Messages,
                    *,
                    stream: bool = False,
                    ) -> Context:
        pass

    @abstractmethod
    async def achat(self,
                    messages: Messages,
                    *,
                    stream: bool = False,
                    ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        pass

    @abstractmethod
    async def aget_context(self,
                           messages: Messages,
                           *,
                           stream: bool = False,
                           ) -> Context:
        pass


class ChatEngine(BaseChatEngine):

    def __init__(self,
                 *,
                 llm: LLM,
                 context_engine: ContextEngine,
                 query_builder: QueryBuilder,
                 max_prompt_tokens: int,
                 max_generated_tokens: int,
                 ):
        self.llm = llm
        self.context_engine = context_engine
        self.query_builder = query_builder
        self.max_prompt_tokens = max_prompt_tokens
        self.max_generated_tokens = max_generated_tokens

    def chat(self,
             messages: Messages,
             *,
             stream: bool = False,
             ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        raise NotImplementedError

    def get_context(self,
                    messages: Messages,
                    *,
                    stream: bool = False,
                    ) -> Context:
        raise NotImplementedError

    async def achat(self,
                    messages: Messages,
                    *,
                    stream: bool = False,
                    ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        raise NotImplementedError

    async def aget_context(self,
                           messages: Messages,
                           *,
                           stream: bool = False,
                           ) -> Context:
        raise NotImplementedError
