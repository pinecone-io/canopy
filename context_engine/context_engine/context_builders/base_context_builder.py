from abc import ABC, abstractmethod
from typing import List

from context_engine.knoweldge_base.models import KBQueryResult
from context_engine.models.data_models import Context


class BaseContextBuilder(ABC):
    """
    BaseContextBuilder is an abstract class that defines the interface for a context builder.
    """

    @abstractmethod
    def build_context(
        self,
        query_results: List[KBQueryResult],
        max_context_tokens: int,
    ) -> Context:
        pass


    @abstractmethod
    async def abuild_context(self,
                             query_results: List[KBQueryResult],
                             max_context_tokens: int,
    ) -> Context:
        pass
