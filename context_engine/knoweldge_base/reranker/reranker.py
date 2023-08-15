from abc import ABC, abstractmethod
from typing import List

from context_engine.knoweldge_base.models import KBQueryResult


class Reranker(ABC):

    @abstractmethod
    def rerank(self, results: List[KBQueryResult]
               ) -> List[KBQueryResult]:
        pass

    @abstractmethod
    async def arerank(self, results: List[KBQueryResult]
                      ) -> List[KBQueryResult]:
        pass


class TransparentReranker(Reranker):
    def rerank(self, results: List[KBQueryResult]
               ) -> List[KBQueryResult]:
        return results

    async def arerank(self, results: List[KBQueryResult]
                      ) -> List[KBQueryResult]:
        return results
