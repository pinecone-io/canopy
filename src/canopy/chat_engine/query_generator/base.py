from abc import ABC, abstractmethod
from typing import List, Optional

from canopy.models.data_models import Messages, Query
from canopy.utils.config import ConfigurableMixin


class QueryGenerator(ABC, ConfigurableMixin):
    @abstractmethod
    def generate(self,
                 messages: Messages,
                 max_prompt_tokens: int,
                 api_key: Optional[str] = None,
                 ) -> List[Query]:
        pass

    @abstractmethod
    async def agenerate(self,
                        messages: Messages,
                        max_prompt_tokens: int,
                        api_key: Optional[str] = None,
                        ) -> List[Query]:
        raise NotImplementedError
