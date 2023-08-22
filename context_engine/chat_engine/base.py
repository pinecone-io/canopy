from abc import ABC, abstractmethod
from typing import Iterable, Union, List

from context_engine.chat_engine.query_builder.base import QueryBuilder
from context_engine.chat_engine.reponse_builder.base import ChatResponseBuilder
from context_engine.context_engine import ContextEngine
from context_engine.llm.base import BaseLLM
from context_engine.models.api_models import StreamingChatResponse, ChatResponse
from context_engine.models.data_models import Context, Messages, Query


class BaseChatEngine(ABC):
    @abstractmethod
    def chat(self,
             messages: Messages,
             *,
             stream: bool = False,
             ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        pass

    @abstractmethod
    def get_context(self, messages: Messages) -> Context:
        pass

    @abstractmethod
    async def achat(self,
                    messages: Messages,
                    *,
                    stream: bool = False,
                    ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        pass

    @abstractmethod
    async def aget_context(self, messages: Messages) -> Context:
        pass


class ChatEngine(BaseChatEngine):

    def __init__(self,
                 *,
                 llm: BaseLLM,
                 query_builder: QueryBuilder,
                 response_builder: ChatResponseBuilder,
                 max_prompt_tokens: int,
                 max_generated_tokens: int,
                 ):
        self.llm = llm
        self.query_builder = query_builder
        self.response_builder = response_builder
        self.max_prompt_tokens = max_prompt_tokens
        self.max_generated_tokens = max_generated_tokens

    def chat(self,
             messages: Messages,
             *,
             stream: bool = False,
             ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        context = self.get_context(messages)
        chat_response = self.response_builder.build(
            context,
            messages,
            max_prompt_tokens=self.max_prompt_tokens,
            stream=stream,
        )
        return chat_response

    def get_query_results(self,
                          messages: Messages,
                          ) -> Context:
        queries = self.query_builder.build(messages,
                                           max_prompt_tokens=self.max_prompt_tokens)
        max_context_tokens = self._calculate_max_context_tokens(self.max_prompt_tokens,
                                                                queries)
        context = self.context_engine.query(queries, max_context_tokens)
        return context

    # TODO: Decide if we want to do this calculation before calling the query builder
    #  based on `messages`, after calling the query builder based on `queries`, or
    #  inside the query builder itself (using pruning_method).
    def _calculate_max_context_tokens(self,
                                      max_prompt_tokens: int,
                                      queries: List[Query]
                                      ) -> int:
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
