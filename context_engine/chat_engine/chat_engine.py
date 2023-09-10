from abc import ABC, abstractmethod
from typing import Iterable, Union, Optional

from context_engine.chat_engine.models import HistoryPruningMethod
from context_engine.chat_engine.prompt_builder import PromptBuilder
from context_engine.chat_engine.query_generator import (QueryGenerator,
                                                        FunctionCallingQueryGenerator, )
from context_engine.context_engine import ContextEngine
from context_engine.knoweldge_base.tokenizer import Tokenizer
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
                 max_context_tokens: Optional[int] = None,
                 query_builder: Optional[QueryGenerator] = None,
                 system_prompt: Optional[str] = None,
                 history_pruning: str = "recent",
                 min_history_messages: int = 1
                 ):
        self.llm = llm
        self.context_engine = context_engine
        self.max_prompt_tokens = max_prompt_tokens
        self.max_generated_tokens = max_generated_tokens
        self._query_builder = query_builder if query_builder is not None else \
            self.DEFAULT_QUERY_GENERATOR(llm=self.llm)
        self.system_prompt_template = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._tokenizer = Tokenizer()
        self._prompt_builder = PromptBuilder(
            history_pruning=HistoryPruningMethod(history_pruning),
            min_history_messages=min_history_messages
        )

        # Set max budget for context tokens, default to 70% of max_prompt_tokens
        max_context_tokens = max_context_tokens or int(max_prompt_tokens * 0.7)
        system_prompt_tokens = self._tokenizer.messages_token_count(
            [SystemMessage(content=self.system_prompt_template)]
        )
        if max_context_tokens + system_prompt_tokens > max_prompt_tokens:
            raise ValueError(
                f"Not enough token budget for knowledge base context. The system prompt"
                f" is taking {system_prompt_tokens} tokens, and together with the "
                f"configured max context tokens {max_context_tokens} it exceeds "
                f"max_prompt_tokens of {self.max_prompt_tokens}"
            )
        self.max_context_tokens = max_context_tokens

    def chat(self,
             messages: Messages,
             *,
             stream: bool = False,
             model_params: Optional[ModelParams] = None
             ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        context = self.get_context(messages)
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

    def get_context(self,
                    messages: Messages,
                    ) -> Context:
        queries = self._query_builder.generate(messages, self.max_prompt_tokens)
        context = self.context_engine.query(queries, self.max_context_tokens)
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
