from abc import ABC, abstractmethod
from typing import List

from context_engine.chat_engine.models import HistoryPrunningMethod
from context_engine.models.data_models import Messages, Query


class QueryBuilder(ABC):

    @abstractmethod
    def build(self,
              messages: Messages,
              max_prompt_tokens: int,
              history_pruning: HistoryPrunningMethod
              ) -> List[Query]:
        pass

    async def abuild(self,
                     messages: Messages,
                     max_prompt_tokens: int,
                     history_pruning: HistoryPrunningMethod
                     ) -> List[Query]:
        raise NotImplementedError
