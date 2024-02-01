import os
from typing import List, Optional


from canopy.knowledge_base.models import KBQueryResult
from canopy.knowledge_base.reranker import Reranker

try:
    import cohere
    from cohere import CohereAPIError
except (OSError, ImportError, ModuleNotFoundError):
    _cohere_installed = False
else:
    _cohere_installed = True


class CohereReranker(Reranker):
    """
    Reranker that uses Cohere's text embedding to rerank documents.

    For each query and documents returned for that query, returns a list
    of documents ordered by their relevance to the provided query.
    """

    def __init__(self,
                 model_name: str = 'rerank-english-v2.0',
                 *,
                 top_n: int = 10,
                 api_key: Optional[str] = None):
        """
            Initializes the Cohere reranker.

            Args:
                model_name: The identifier of the model to use, one of :
                    rerank-english-v2.0, rerank-multilingual-v2.0
                top_n: The number of most relevant documents return, defaults to 10
                api_key: API key for Cohere. If not passed `CO_API_KEY` environment
                    variable will be used.
        """

        if not _cohere_installed:
            raise ImportError(
                "Failed to import cohere. Make sure you install cohere extra "
                "dependencies by running: "
                "pip install canopy-sdk[cohere]"
            )
        cohere_api_key = api_key or os.environ.get("CO_API_KEY")
        if cohere_api_key is None:
            raise RuntimeError(
                "Cohere API key is required to use Cohere Reranker. "
                "Please provide it as an argument "
                "or set the CO_API_KEY environment variable."
            )
        self._client = cohere.Client(api_key=cohere_api_key)
        self._model_name = model_name
        self._top_n = top_n

    def rerank(self, results: List[KBQueryResult]) -> List[KBQueryResult]:
        reranked_query_results: List[KBQueryResult] = []
        for result in results:
            texts = [doc.text for doc in result.documents]
            try:
                response = self._client.rerank(query=result.query,
                                               documents=texts,
                                               top_n=self._top_n,
                                               model=self._model_name)
            except CohereAPIError as e:
                raise RuntimeError("Failed to rerank documents using Cohere."
                                   f" Underlying Error:\n{e.message}")

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
