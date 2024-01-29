import os

import pytest

from canopy.knowledge_base.models import KBQueryResult, KBDocChunkWithScore
from canopy.knowledge_base.reranker import CohereReranker
from canopy.models.data_models import Query


@pytest.fixture
def cohere_reranker():
    if os.getenv("CO_API_KEY") is None:
        pytest.skip(
            "Couldn't find Cohere API key. Skipping Cohere tests."
        )
    return CohereReranker()


def test_rerank_empty(cohere_reranker):
    results = cohere_reranker.rerank([])
    assert results == []


def test_rerank_ranks(cohere_reranker):
    documents = [
        KBDocChunkWithScore(
            id=f"doc_1_{i}",
            text=f"Sample chunk {i}",
            document_id="doc_1",
            source="doc_1",
            score=0.1 * i
        ) for i in range(4)
    ]
    query_result = KBQueryResult(query="Sample query 1",
                                 documents=documents)
    ranked_result = next(iter(cohere_reranker.rerank([query_result])))
    scores = [doc.score for doc in ranked_result.documents]

    assert len(ranked_result.documents) == len(documents)
    assert scores == sorted(scores, reverse=True)
