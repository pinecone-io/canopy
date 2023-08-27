from abc import ABC, abstractmethod
from typing import List

from context_engine.models.data_models import Messages, Query


class QueryGenerator(ABC):

    @abstractmethod
    def generate(self,
                 messages: Messages,
                 max_prompt_tokens: int,
                 ) -> List[Query]:
        pass

    @abstractmethod
    async def agenerate(self,
                        messages: Messages,
                        max_prompt_tokens: int,
                        ) -> List[Query]:
        raise NotImplementedError
