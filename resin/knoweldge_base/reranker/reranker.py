from abc import ABC, abstractmethod
from typing import List

from resin.knoweldge_base.models import KBQueryResult
from resin.utils.config import ConfigurableMixin


class Reranker(ABC, ConfigurableMixin):

    @abstractmethod
    def rerank(self, results: List[KBQueryResult]) -> List[KBQueryResult]:
        pass

    @abstractmethod
    async def arerank(self, results: List[KBQueryResult]) -> List[KBQueryResult]:
        pass


class TransparentReranker(Reranker):
    def rerank(self, results: List[KBQueryResult]) -> List[KBQueryResult]:
        return results

    async def arerank(self, results: List[KBQueryResult]) -> List[KBQueryResult]:
        return results
