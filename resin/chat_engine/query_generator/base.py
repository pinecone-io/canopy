from abc import ABC, abstractmethod
from typing import List

from resin.models.data_models import Messages, Query
from resin.utils.config import ConfigurableMixin


class QueryGenerator(ABC, ConfigurableMixin):
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
