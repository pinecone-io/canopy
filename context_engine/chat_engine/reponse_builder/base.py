from abc import ABC, abstractmethod
from typing import Union, Iterable, List

from context_engine.chat_engine.history_builder.base_history_builder\
    import BaseHistoryBuilder
from context_engine.context_engine.context_builder.base import BaseContextBuilder
from context_engine.knoweldge_base.models import QueryResult
from context_engine.models.api_models import ChatResponse, StreamingChatResponse
from context_engine.models.data_models import Context, Messages


class ChatResponseBuilder(ABC):

    def __init__(
        self,
        context_builder: BaseContextBuilder,
        history_builder: BaseHistoryBuilder,
        context_ratio: float = 0.5,
        max_prompt_tokens: int = 4096,
    ):
        self.context_builder = context_builder
        self.history_builder = history_builder
        self.context_ratio = context_ratio
        self.max_prompt_tokens = max_prompt_tokens

    def build(self,
              query_results: List[QueryResult],
              messages: Messages,
              max_prompt_tokens: int,
              stream: bool = False,
              ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        history = self.history_builder.build(messages,
                                             int(max_prompt_tokens
                                                 * self.context_ratio))
        context = self.context_builder.build(query_results,
                                             max_prompt_tokens - history.num_tokens)  # type: ignore
        return self.merge(context, history)  # type: ignore

    async def abuild(self,
                     context: Context,
                     messages: Messages,
                     max_prompt_tokens: int,
                     stream: bool = False,
                     ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        raise NotImplementedError

    @abstractmethod
    def merge(
        self,
        context: Context,
        history: Context,  # TODO: is this some other object?
    ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        pass
