from abc import ABC, abstractmethod
from typing import Iterable, Union, Optional

from context_engine.chat_engine.models import HistoryPruningMethod
from context_engine.chat_engine.prompt_builder import PromptBuilder
from context_engine.chat_engine.query_generator import (QueryGenerator,
                                                        FunctionCallingQueryGenerator, )
from context_engine.context_engine import ContextEngine
from context_engine.knoweldge_base.tokenizer.base import Tokenizer
from context_engine.llm import BaseLLM
from context_engine.llm.models import ModelParams, SystemMessage
from context_engine.models.api_models import StreamingChatResponse, ChatResponse
from context_engine.models.data_models import Context, Messages


DEFAULT_SYSTEM_PROMPT = """"Use the following pieces of context to answer the user question at the next messages. This context retrieved from a knowledge database and you should use only the facts from the context to answer. Always remember to include the reference to the documents you used from their 'reference' field in the format 'Source: $REFERENCE_HERE'.
If you don't know the answer, just say that you don't know, don't try to make up an answer, use the context."
Don't address the context directly, but use it to answer the user question like it's your own knowledge."""  # noqa


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

    DEFAULT_QUERY_GENERATOR = FunctionCallingQueryGenerator

    def __init__(self,
                 *,
                 llm: BaseLLM,
                 context_engine: ContextEngine,
                 max_prompt_tokens: int,
                 max_generated_tokens: int,
                 query_builder: Optional[QueryGenerator] = None,
                 system_prompt: Optional[str] = None,
                 context_to_history_ratio: float = 0.8,
                 history_pruning: str = "recent",
                 min_history_messages: int = 3
                 ):
        self.system_prompt_template = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.llm = llm
        self.context_engine = context_engine
        self.max_prompt_tokens = max_prompt_tokens
        self.max_generated_tokens = max_generated_tokens
        self._query_builder = query_builder if query_builder is not None else \
            self.DEFAULT_QUERY_GENERATOR(llm=self.llm)
        self._context_to_history_ratio = context_to_history_ratio
        self._tokenizer = Tokenizer()
        self._prompt_builder = PromptBuilder(
            history_pruning=HistoryPruningMethod(history_pruning),
            min_history_messages=min_history_messages
        )

    def chat(self,
             messages: Messages,
             *,
             stream: bool = False,
             model_params: Optional[ModelParams] = None
             ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        queries = self._query_builder.generate(messages,
                                               max_prompt_tokens=self.max_prompt_tokens)

        max_context_tokens = self._calculate_max_context_tokens(messages)
        context = self.context_engine.query(queries, max_context_tokens)

        system_prompt = self.system_prompt_template + f"\nContext: {context.to_text()}"
        llm_messages = self._prompt_builder.build(
            system_prompt,
            messages,
            max_tokens=self.max_prompt_tokens
        )
        return self.llm.chat_completion(llm_messages,
                                        max_tokens=self.max_generated_tokens,
                                        stream=stream,
                                        model_params=model_params)

    def _calculate_max_context_tokens(self, messages: Messages):
        history_tokens = self._tokenizer.messages_token_count(messages)
        max_context_tokens = max(
            self.max_prompt_tokens - history_tokens,
            int(self.max_prompt_tokens * self._context_to_history_ratio)
        )

        system_prompt_tokens = self._tokenizer.messages_token_count(
            [SystemMessage(content=self.system_prompt_template)]
        )
        max_context_tokens -= system_prompt_tokens
        if max_context_tokens <= 0:
            raise ValueError(f"Not enough token budget for generating context. The "
                             f"prunned history is taking {history_tokens} tokens, "
                             f"and the system prompt is taking {system_prompt_tokens} "
                             f"tokens, which is more than the max prompt tokens "
                             f"{self.max_prompt_tokens}")

        return max_context_tokens

    def get_context(self,
                    messages: Messages,
                    ) -> Context:
        queries = self._query_builder.generate(messages,
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
