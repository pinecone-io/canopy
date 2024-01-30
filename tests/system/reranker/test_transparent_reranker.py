import pytest

from canopy.knowledge_base.models import KBDocChunkWithScore, KBQueryResult
from canopy.knowledge_base.reranker import TransparentReranker


@pytest.fixture
def documents():
    return [
        KBDocChunkWithScore(
            id=f"doc_1_{i}",
            text=f"Sample chunk {i}",
            document_id="doc_1",
            source="doc_1",
            score=0.1 * i
        ) for i in range(1)
    ]


@pytest.fixture
def query_result(documents):
    return KBQueryResult(query="Sample query 1",
                         documents=documents)


def test_rerank(query_result):
    assert TransparentReranker().rerank([query_result]) == [query_result]
