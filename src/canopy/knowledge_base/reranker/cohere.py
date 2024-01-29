from typing import List

from canopy.knowledge_base.models import KBQueryResult
from canopy.knowledge_base.reranker import Reranker

try:
    import cohere
except (OSError, ImportError, ModuleNotFoundError) as e:
    _cohere_installed = False
else:
    _cohere_installed = True


class CohereReranker(Reranker):
    """
    Reranker that uses Cohere's text embedding to rerank documents.

    Note: You should provide an API key as the environment variable `CO_API_KEY`.
    """

    def __init__(self,
                 model_name: str = 'rerank-english-v2.0',
                 top_n: int = 10):

        if not _cohere_installed:
            raise ImportError(
                "Failed to import cohere. Make sure you install cohere extra "
                "dependencies by running: "
                "pip install canopy-sdk[cohere]"
            )
        self._client = cohere.Client()
        self._model_name = model_name
        self._top_n = top_n

    def rerank(self, results: List[KBQueryResult]) -> List[KBQueryResult]:
        reranked_query_results: List[KBQueryResult] = []
        for result in results:
            texts = [doc.text for doc in result.documents]
            response = self._client.rerank(query=result.query,
                                           documents=texts,
                                           top_n=self._top_n,
                                           model=self._model_name)
            reranked_docs = []
            for rerank_result in response:
                doc = result.documents[rerank_result.index].copy(
                    deep=True,
                    update=dict(score=rerank_result.relevance_score)
                )
                reranked_docs.append(doc)

            reranked_query_results.append(KBQueryResult(query=result.query,
                                                        documents=reranked_docs))
        return reranked_query_results

    async def arerank(self, results: List[KBQueryResult]) -> List[KBQueryResult]:
        raise NotImplementedError()
