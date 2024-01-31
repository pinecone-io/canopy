from typing import List

from canopy.knowledge_base.models import KBQueryResult
from canopy.knowledge_base.reranker import Reranker


class TransparentReranker(Reranker):
    """
    Transparent reranker that does nothing, it just returns the results as is. This is the default reranker.
    The TransparentReranker is used as a placeholder for future development "forcing" every result set to be reranked.
    """  # noqa: E501

    def rerank(self, results: List[KBQueryResult]) -> List[KBQueryResult]:
        """
        Returns the results as is.

        Args:
            results: A list of KBQueryResult to rerank.

        Returns:
            results: A list of KBQueryResult, same as the input.
        """  # noqa: E501
        return results

    async def arerank(self, results: List[KBQueryResult]) -> List[KBQueryResult]:
        return results
