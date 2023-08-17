from abc import ABC, abstractmethod
from typing import List

from context_engine.models.data_models import Messages, Query


class QueryBuilder(ABC):

    @abstractmethod
    def build(self,
              messages: Messages,
              max_prompt_tokens: int,
              ) -> List[Query]:
        pass

    @abstractmethod
    async def abuild(self,
                     messages: Messages,
                     max_prompt_tokens: int,
                     ) -> List[Query]:
        raise NotImplementedError
