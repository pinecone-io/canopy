from unittest.mock import patch

import pytest

from canopy.knowledge_base.models import KBDocChunk
from canopy.knowledge_base.record_encoder import HybridRecordEncoder
from canopy.models.data_models import Query
from tests.unit.stubs.stub_dense_encoder import StubDenseEncoder


@pytest.fixture
def documents():
    return [KBDocChunk(id=f"doc_1_{i}",
                       text=f"Sample document {i}",
                       document_id=f"doc_{i}",
                       metadata={"test": i},
                       source="doc_1")
            for i in range(5)]


@pytest.fixture
def queries():
    return [Query(text="Sample query 1"),
            Query(text="Sample query 2"),
            Query(text="Sample query 3")]


@pytest.fixture
def inner_dimension():
    return 4


@pytest.fixture
def inner_encoder(inner_dimension):
    return StubDenseEncoder(dimension=inner_dimension)


@pytest.fixture
def encoder(inner_encoder):
    return HybridRecordEncoder(inner_encoder, batch_size=2)


def test_dimension(encoder, inner_dimension):
    assert encoder.dimension == inner_dimension


def test_init_encoder_invalid_alpha(inner_encoder):
    with pytest.raises(ValueError):
        HybridRecordEncoder(inner_encoder, alpha=-1)
    with pytest.raises(ValueError):
        HybridRecordEncoder(inner_encoder, alpha=2)


def test_encode_documents(encoder, documents, queries):
    encoded_documents = encoder.encode_documents(documents)
    for encoded_document in encoded_documents:
        assert len(encoded_document.values) == encoder.dimension
        assert len(encoded_document.sparse_values) > 0


def test_encode_queries(encoder, queries):
    encoded_queries = encoder.encode_queries(queries)
    assert len(encoded_queries) == len(queries)


def test_encode_queries_with_alpha_1(encoder, inner_encoder, queries):
    """
        Tests whether the encoded queries are exactly the same as the dense
        encoded queries when alpha is 1.
    """
    with patch.object(encoder, '_alpha', new=1.0):
        encoded_queries = encoder.encode_queries(queries)
        dense_queries = inner_encoder.encode_queries([q.text for q in queries])

        assert len(encoded_queries) == len(dense_queries)
        for encoded_query, dense_query in zip(encoded_queries, dense_queries):
            assert encoded_query.values == dense_query


def test_encode_queries_with_alpha_0(encoder, inner_encoder, queries):
    """
        Tests whether the encoded queries are exactly the same as the sparse
        encoded queries when alpha is 0.
    """
    with patch.object(encoder, '_alpha', new=0.0):
        encoded_queries = encoder.encode_queries(queries)
        sparse_encoder = encoder._sparse_encoder
        sparse_queries = sparse_encoder.encode_queries([q.text for q in queries])

        assert len(encoded_queries) == len(sparse_queries)
        for encoded_query, sparse_query in zip(encoded_queries, sparse_queries):
            assert encoded_query.sparse_values == sparse_query
