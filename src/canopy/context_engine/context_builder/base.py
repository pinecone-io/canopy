from abc import ABC, abstractmethod
from typing import List

from src.canopy.knowledge_base.models import QueryResult
from src.canopy.models.data_models import Context
from src.canopy.utils.config import ConfigurableMixin


class ContextBuilder(ABC, ConfigurableMixin):
    """
    BaseContextBuilder is an abstract class that defines the interface for a context
    builder.
    """

    @abstractmethod
    def build(self,
              query_results: List[QueryResult],
              max_context_tokens: int, ) -> Context:
        pass

    @abstractmethod
    async def abuild(self,
                     query_results: List[QueryResult],
                     max_context_tokens: int, ) -> Context:
        pass
