from abc import ABC, abstractmethod
from typing import List

from context_engine.knoweldge_base.models import QueryResult
from context_engine.models.data_models import Context


class ContextBuilder(ABC):
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
