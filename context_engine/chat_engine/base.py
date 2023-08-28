from abc import ABC, abstractmethod
from typing import Iterable, Union, Optional

from context_engine.chat_engine.prompt_builder.base import BasePromptBuilder
from context_engine.chat_engine.query_generator.base import QueryGenerator
from context_engine.knoweldge_base import KnowledgeBase
from context_engine.llm.base import BaseLLM
from context_engine.llm.models import ModelParams
from context_engine.models.api_models import StreamingChatResponse, ChatResponse
from context_engine.models.data_models import Context, Messages


class BaseChatEngine(ABC):
    @abstractmethod
    def chat(self,
             messages: Messages,
             *,
             stream: bool = False,
             model_params: Optional[ModelParams] = None
             ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        pass

    # TODO: Decide if we want it for first release in the API
    @abstractmethod
    def get_context(self, messages: Messages) -> Context:
        pass

    @abstractmethod
    async def achat(self,
                    messages: Messages,
                    *,
                    stream: bool = False,
                    model_params: Optional[ModelParams] = None
                    ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        pass

    @abstractmethod
    async def aget_context(self, messages: Messages) -> Context:
        pass


class ChatEngine(BaseChatEngine):

    def __init__(self,
                 *,
                 system_message: str,
                 llm: BaseLLM,
                 query_builder: QueryGenerator,
                 knowledge_base: KnowledgeBase,
                 prompt_builder: BasePromptBuilder,
                 max_prompt_tokens: int,
                 max_generated_tokens: int,
                 ):
        self.system_message = system_message
        self.llm = llm
        self.query_builder = query_builder
        self.knowledge_base = knowledge_base
        self.prompt_builder = prompt_builder
        self.max_prompt_tokens = max_prompt_tokens
        self.max_generated_tokens = max_generated_tokens

    def chat(self,
             messages: Messages,
             *,
             stream: bool = False,
             model_params: Optional[ModelParams] = None
             ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        queries = self.query_builder.generate(messages,
                                              max_prompt_tokens=self.max_prompt_tokens)
        query_results = self.knowledge_base.query(queries)
        prompt_messages = self.prompt_builder.build(self.system_message,
                                                    messages,
                                                    query_results,
                                                    max_tokens=self.max_prompt_tokens)
        return self.llm.chat_completion(prompt_messages,
                                        max_tokens=self.max_generated_tokens,
                                        stream=stream,
                                        model_params=model_params)

    def get_context(self,
                    messages: Messages,
                    ) -> Context:
        raise NotImplementedError

    async def achat(self,
                    messages: Messages,
                    *,
                    stream: bool = False,
                    model_params: Optional[ModelParams] = None
                    ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        raise NotImplementedError

    @abstractmethod
    async def aget_context(self, messages: Messages) -> Context:
        raise NotImplementedError
