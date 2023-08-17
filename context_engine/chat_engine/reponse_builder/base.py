from abc import ABC, abstractmethod
from typing import Union, Iterable

from context_engine.chat_engine.models import HistoryPruningMethod
from context_engine.models.api_models import ChatResponse, StreamingChatResponse
from context_engine.models.data_models import Context, Messages


class ChatResponseBuilder(ABC):
    @abstractmethod
    def build(self,
              context: Context,
              messages: Messages,
              max_prompt_tokens: int,
              history_pruning: HistoryPruningMethod
              ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        pass

    async def abuild(self,
                     context: Context,
                     messages: Messages,
                     max_prompt_tokens: int,
                     history_pruning: HistoryPruningMethod
                     ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        raise NotImplementedError
