import os

import pytest

from canopy.knowledge_base.models import KBQueryResult, KBDocChunkWithScore
from canopy.knowledge_base.reranker import CohereReranker


@pytest.fixture
def should_run_test():
    if os.getenv("CO_API_KEY") is None:
        pytest.skip(
            "Couldn't find Cohere API key. Skipping Cohere tests."
        )


@pytest.fixture
def cohere_reranker(should_run_test):
    return CohereReranker()


@pytest.fixture
def documents():
    return [
        KBDocChunkWithScore(
            id=f"doc_1_{i}",
            text=f"Sample chunk {i}",
            document_id="doc_1",
            source="doc_1",
            score=0.1 * i
        ) for i in range(4)
    ]


@pytest.fixture
def query_result(documents):
    return KBQueryResult(query="Sample query 1",
                         documents=documents)


def test_rerank_empty(cohere_reranker):
    results = cohere_reranker.rerank([])
    assert results == []


def test_rerank(cohere_reranker, query_result, documents):
    ranked_result = next(iter(cohere_reranker.rerank([query_result])))
    scores = [doc.score for doc in ranked_result.documents]

    assert len(ranked_result.documents) == len(documents)
    assert scores == sorted(scores, reverse=True)


def test_bad_api_key(should_run_test, query_result):
    from cohere import CohereAPIError
    with pytest.raises(CohereAPIError, match="invalid api token"):
        CohereReranker(api_key="bad key").rerank([query_result])


def test_top_n(should_run_test, query_result):
    results = CohereReranker(top_n=1).rerank([query_result])
    assert len(results[0].documents) == 1
