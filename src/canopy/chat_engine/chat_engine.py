import os
from abc import ABC, abstractmethod
from typing import Iterable, Union, Optional, cast

from canopy.chat_engine.models import HistoryPruningMethod
from canopy.chat_engine.prompt_builder import PromptBuilder
from canopy.chat_engine.query_generator import (QueryGenerator,
                                                FunctionCallingQueryGenerator, )
from canopy.context_engine import ContextEngine
from canopy.tokenizer import Tokenizer
from canopy.llm import BaseLLM, OpenAILLM
from canopy.models.api_models import (StreamingChatChunk, ChatResponse,
                                      StreamingChatResponse, )
from canopy.models.data_models import Context, Messages, SystemMessage
from canopy.utils.config import ConfigurableMixin

CE_DEBUG_INFO = os.getenv("CE_DEBUG_INFO", "FALSE").lower() == "true"


DEFAULT_SYSTEM_PROMPT = """Use the following pieces of context to answer the user question at the next messages. This context retrieved from a knowledge database and you should use only the facts from the context to answer. Always remember to include the source to the documents you used from their 'source' field in the format 'Source: $SOURCE_HERE'.
If you don't know the answer, just say that you don't know, don't try to make up an answer, use the context.
Don't address the context directly, but use it to answer the user question like it's your own knowledge.
"""  # noqa


class BaseChatEngine(ABC, ConfigurableMixin):
    @abstractmethod
    def chat(self,
             messages: Messages,
             *,
             stream: bool = False,
             model_params: Optional[dict] = None
             ) -> Union[ChatResponse, StreamingChatResponse]:
        pass

    @abstractmethod
    async def achat(self,
                    messages: Messages,
                    *,
                    stream: bool = False,
                    model_params: Optional[dict] = None
                    ) -> Union[ChatResponse, StreamingChatResponse]:
        pass

    @abstractmethod
    async def aget_context(self, messages: Messages) -> Context:
        pass


class ChatEngine(BaseChatEngine):

    """
    Chat engine is an object that implements end to end chat API with [RAG](https://www.pinecone.io/learn/retrieval-augmented-generation/).
    Given chat history, the chat engine orchestrates its underlying context engine and LLM to run the following steps:

    1. Generate search queries from the chat history
    2. Retrieve the most relevant context for each query using the context engine
    3. Prompt the LLM with the chat history and the retrieved context to generate the next response

    To use the chat engine, you need to provide it with a context engine, and optionally you can configure all its other components such as the LLM, the query generator, and the prompt builder and more.

    Example:

        >>> from canopy.chat_engine import ChatEngine
        >>> chat_engine = ChatEngine(context_engine)

    Where you can follow the instructions in the [context engine](../context_engine/context_engine) to create a context engine.
    Then you can use the chat engine to chat with a user:

        >>> from canopy.models.data_models import UserMessage
        >>> messages = [UserMessage(content="Hello! what is the capital of France?")]
        >>> response = chat_engine.chat(messages)
        >>> print(response.choices[0].message.content)
        Paris is the capital of France. Source: https://en.wikipedia.org/wiki/Paris
    """  # noqa: E501

    _DEFAULT_COMPONENTS = {
        'context_engine': ContextEngine,
        'llm': OpenAILLM,
        'query_builder': FunctionCallingQueryGenerator,
    }

    def __init__(self,
                 context_engine: ContextEngine,
                 *,
                 llm: Optional[BaseLLM] = None,
                 max_prompt_tokens: int = 4096,
                 max_generated_tokens: Optional[int] = None,
                 max_context_tokens: Optional[int] = None,
                 query_builder: Optional[QueryGenerator] = None,
                 system_prompt: Optional[str] = None,
                 history_pruning: str = "recent",
                 min_history_messages: int = 1
                 ):
        """
        Initialize a chat engine.

        Args:
            context_engine: An instance of a context engine to use for retrieving context to prompt the LLM along with the chat history.
            llm: An instance of a LLM to use for generating the next response. Defaults to OpenAILLM.
            max_prompt_tokens: The maximum number of tokens to use for the prompt to the LLM. Defaults to 4096.
            max_generated_tokens: The maximum number of tokens to generate from the LLM. Defaults to None, which means the LLM will use its default behavior.
            max_context_tokens: The maximum number of tokens to use for the context to prompt the LLM. Defaults to be 70% of the max_prompt_tokens.
            query_builder: An instance of a query generator to use for generating queries from the chat history. Defaults to FunctionCallingQueryGenerator.
            system_prompt: The system prompt to use for the LLM. Defaults to a generic prompt that is suitable for most use cases.
            history_pruning: The history pruning method to use for truncating the chat history to a prompt. Defaults to "recent", which means the chat history will be truncated to the most recent messages.
            min_history_messages: The minimum number of messages to keep in the chat history. Defaults to 1.
        """  # noqa: E501
        if not isinstance(context_engine, ContextEngine):
            raise TypeError(
                f"context_engine must be an instance of ContextEngine, "
                f"got {type(context_engine)}"
            )
        self.context_engine = context_engine

        if llm:
            if not isinstance(llm, BaseLLM):
                raise TypeError(
                    f"llm must be an instance of BaseLLM, got {type(llm)}"
                )
            self.llm = llm
        else:
            self.llm = self._DEFAULT_COMPONENTS['llm']()

        if query_builder:
            if not isinstance(query_builder, QueryGenerator):
                raise TypeError(
                    f"query_builder must be an instance of QueryGenerator, "
                    f"got {type(query_builder)}"
                )
            self._query_builder = query_builder
        else:
            self._query_builder = self._DEFAULT_COMPONENTS['query_builder']()

        self.max_prompt_tokens = max_prompt_tokens
        self.max_generated_tokens = max_generated_tokens
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
             model_params: Optional[dict] = None
             ) -> Union[ChatResponse, StreamingChatResponse]:
        """
        Chat completion with RAG. Given a list of messages (history), the chat engine will generate the next response, based on the relevant context retrieved from the knowledge base.

        While calling the chat method, behind the scenes the chat engine will do the following:
        1. Generate search queries from the chat history
        2. Retrieve the most relevant context for each query using the context engine
        3. Prompt the LLM with the chat history and the retrieved context to generate the next response
        4. Return the response

        Args:
            messages: A list of messages (history) to generate the next response from.
            stream: A boolean flag to indicate if the chat should be streamed or not. Defaults to False.
            model_params: A dictionary of model parameters to use for the LLM. Defaults to None, which means the LLM will use its default values.

        Returns:
            A ChatResponse object if stream is False, or a StreamingChatResponse object if stream is True.

        Examples:

            >>> from canopy.models.data_models import UserMessage
            >>> messages = [UserMessage(content="Hello! what is the capital of France?")]
            >>> response = chat_engine.chat(messages)
            >>> print(response.choices[0].message.content)

            Or you can stream the response:
            >>> response = chat_engine.chat(messages, stream=True)
            >>> for chunk in response.chunks:
            ...     print(chunk.json())
        """  # noqa: E501
        context = self._get_context(messages)
        system_prompt = self.system_prompt_template + f"\nContext: {context.to_text()}"
        llm_messages = self._prompt_builder.build(
            system_prompt,
            messages,
            max_tokens=self.max_prompt_tokens
        )
        llm_response = self.llm.chat_completion(llm_messages,
                                                max_tokens=self.max_generated_tokens,
                                                stream=stream,
                                                model_params=model_params)
        debug_info = {}
        if CE_DEBUG_INFO:
            debug_info['context'] = context.dict()
            debug_info['context'].update(context.debug_info)

        if stream:
            return StreamingChatResponse(
                chunks=cast(Iterable[StreamingChatChunk], llm_response),
                debug_info=debug_info
            )
        else:
            response = cast(ChatResponse, llm_response)
            response.debug_info = debug_info
            return response

    def _get_context(self,
                     messages: Messages,
                     ) -> Context:
        queries = self._query_builder.generate(messages, self.max_prompt_tokens)
        context = self.context_engine.query(queries, self.max_context_tokens)
        return context

    async def achat(self,
                    messages: Messages,
                    *,
                    stream: bool = False,
                    model_params: Optional[dict] = None
                    ) -> Union[ChatResponse, StreamingChatResponse]:
        raise NotImplementedError

    async def aget_context(self, messages: Messages) -> Context:
        raise NotImplementedError
