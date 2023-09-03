from abc import ABC, abstractmethod
from typing import Iterable, Union, Optional

from context_engine.chat_engine.prompt_builder.base import PromptBuilder
from context_engine.context_engine import ContextEngine
from context_engine.chat_engine.query_generator.base import QueryGenerator
from context_engine.knoweldge_base import KnowledgeBase
from context_engine.knoweldge_base.tokenizer.base import Tokenizer
from context_engine.llm.base import BaseLLM
from context_engine.llm.models import ModelParams, UserMessage
from context_engine.models.api_models import StreamingChatResponse, ChatResponse
from context_engine.models.data_models import Context, Messages
from context_engine.chat_engine.history_builder import (RecentHistoryBuilder,
                                                        RaisingHistoryBuilder, )

DEFAULT_SYSTEM_PROMPT = ""  #TODO


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
                 llm: BaseLLM,
                 context_engine: ContextEngine,
                 query_builder: QueryGenerator,
                 max_prompt_tokens: int,
                 max_generated_tokens: int,
                 tokenizer: Tokenizer,  # TODO: Remove this dependency
                 system_prompt: Optional[str] = None,
                 context_to_history_ratio: int = 0.8
                 ):
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.llm = llm
        self.context_engine = context_engine
        self.query_builder = query_builder
        self.max_prompt_tokens = max_prompt_tokens
        self.max_generated_tokens = max_generated_tokens
        self._context_to_history_ratio = context_to_history_ratio

        # TODO: hardcoded for now, need to make it configurable
        history_prunner = RaisingHistoryBuilder(tokenizer)
        self._prompt_builder = PromptBuilder(tokenizer, history_prunner)

    def chat(self,
             messages: Messages,
             *,
             stream: bool = False,
             model_params: Optional[ModelParams] = None
             ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        queries = self.query_builder.generate(messages,
                                              max_prompt_tokens=self.max_prompt_tokens)

        # Before calling the ContextEngine, build the prompt for the LLM completion.
        # This already applies the history trimming policy (if required) - guaranteeing
        # that enough tokens are reserved for the Context.
        llm_messages, history_tokens = self._prompt_builder.build(
            self.system_prompt,
            messages,
            max_tokens=int(
                self.max_prompt_tokens * (1 - self._context_to_history_ratio)
            )
        )
        max_context_tokens = self.max_prompt_tokens - history_tokens
        context = self.context_engine.query(queries, max_context_tokens)
        llm_messages.append(UserMessage(content=context.content.to_text()))

        return self.llm.chat_completion(llm_messages,
                                        max_tokens=self.max_generated_tokens,
                                        stream=stream,
                                        model_params=model_params)

    def get_context(self,
                    messages: Messages,
                    ) -> Context:
        queries = self.query_builder.generate(messages,
                                              max_prompt_tokens=self.max_prompt_tokens)

        context = self.context_engine.query(queries,
                                            max_context_tokens=self.max_prompt_tokens)
        return context

    async def achat(self,
                    messages: Messages,
                    *,
                    stream: bool = False,
                    model_params: Optional[ModelParams] = None
                    ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        raise NotImplementedError

    async def aget_context(self, messages: Messages) -> Context:
        raise NotImplementedError
