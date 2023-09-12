import logging
from abc import ABC, abstractmethod
from typing import Iterable, Union, Optional

from context_engine.chat_engine.models import HistoryPruningMethod
from context_engine.chat_engine.prompt_builder import PromptBuilder
from context_engine.chat_engine.query_generator import (QueryGenerator,
                                                        QUERY_GENERATOR_CLASSES, )
from context_engine.context_engine import ContextEngine
from context_engine.knoweldge_base.tokenizer import Tokenizer
from context_engine.llm import BaseLLM
from context_engine.llm.models import ModelParams, SystemMessage
from context_engine.models.api_models import StreamingChatResponse, ChatResponse
from context_engine.models.data_models import Context, Messages
from context_engine.utils import initialize_from_config


logger = logging.getLogger(__name__)
DEFAULT_SYSTEM_PROMPT = """"Use the following pieces of context to answer the user question at the next messages. This context retrieved from a knowledge database and you should use only the facts from the context to answer. Always remember to include the source to the documents you used from their 'source' field in the format 'Source: $SOURCE_HERE'.
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
        if not isinstance(llm, BaseLLM):
            raise ValueError(f"llm must be an instance of BaseLLM, got {type(llm)}")
        self.llm = llm

        if not isinstance(context_engine, ContextEngine):
            raise ValueError(f"context_engine must be an instance of ContextEngine,"
                             f" got {type(context_engine)}")
        self.context_engine = context_engine

        self.max_prompt_tokens = max_prompt_tokens
        self.max_generated_tokens = max_generated_tokens
        self.system_prompt_template = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._tokenizer = Tokenizer()
        self._prompt_builder = PromptBuilder(
            history_pruning=HistoryPruningMethod(history_pruning),
            min_history_messages=min_history_messages
        )

        if query_builder is None:
            query_builder_type = QUERY_GENERATOR_CLASSES['default']
            logger.info(f"Initializing ChatEngine with default query builder: "
                        f"{query_builder_type.__name__}")
            self._query_builder = query_builder_type(llm=self.llm)
        else:
            if not isinstance(query_builder, QueryGenerator):
                raise ValueError(f"query_builder must be an instance of QueryGenerator,"
                                 f" got {type(query_builder)}")
            self._query_builder = query_builder

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

    @classmethod
    def from_config(cls,
                    config: dict,
                    *,
                    llm: BaseLLM,
                    context_engine: ContextEngine,
                    query_builder: Optional[QueryGenerator] = None,
                    ):
        unallowed_keys = set(config.keys()).intersection({'llm', 'context_engine'})
        if unallowed_keys:
            raise ValueError(f"Unallowed keys in ChatEngine config: {unallowed_keys}")

        query_builder_cfg = config.pop('query_builder', None)
        if query_builder and query_builder_cfg:
            raise ValueError("Cannot provide both query_builder override and "
                             "query_builder config. If you wish to override with your"
                             " own query_builder, remove the 'query_builder' "
                             "key from the config")
        if query_builder_cfg:
            query_builder = initialize_from_config(query_builder_cfg,
                                                   QUERY_GENERATOR_CLASSES,
                                                   "query_builder")

        return cls(llm=llm,
                   context_engine=context_engine,
                   query_builder=query_builder,
                   **config)

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
