from abc import ABC, abstractmethod
from typing import List

from resin.knoweldge_base.models import QueryResult
from resin.models.data_models import Context
from resin.utils.config import FactoryMixin


class ContextBuilder(ABC, FactoryMixin):
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
