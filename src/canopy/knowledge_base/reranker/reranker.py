from abc import ABC, abstractmethod
from typing import List

from canopy.knowledge_base.models import KBQueryResult
from canopy.utils.config import ConfigurableMixin


class Reranker(ABC, ConfigurableMixin):
    """
    Abstract class for rerankers. Rerankers take a list of KBQueryResult and return a list of KBQueryResult,
    where the results are reranked according to the reranker logic.
    Reranker is an abstract class that must be subclassed to be used,
    """  # noqa: E501

    @abstractmethod
    def rerank(self, results: List[KBQueryResult]) -> List[KBQueryResult]:
        pass

    @abstractmethod
    async def arerank(self, results: List[KBQueryResult]) -> List[KBQueryResult]:
        pass
