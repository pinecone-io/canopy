import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pinecone_text.sparse import BM25Encoder

from canopy.knowledge_base.models import KBDocChunk
from canopy.knowledge_base.record_encoder import HybridRecordEncoder
from canopy.models.data_models import Query
from tests.unit.stubs.stub_dense_encoder import StubDenseEncoder


@pytest.fixture(scope="module")
def documents():
    return [KBDocChunk(id=f"doc_1_{i}",
                       text=f"Sample document {i}",
                       document_id=f"doc_{i}",
                       metadata={"test": i},
                       source="doc_1")
            for i in range(5)]


@pytest.fixture(scope="module")
def queries():
    return [Query(text="Sample query 1"),
            Query(text="Sample query 2"),
            Query(text="Sample query 3")]


@pytest.fixture(scope="module")
def inner_dimension():
    return 4


@pytest.fixture(scope="module")
def inner_encoder(inner_dimension):
    return StubDenseEncoder(dimension=inner_dimension)


@pytest.fixture(scope="module")
def bm_25_encoder_df_path(documents):
    bm25 = BM25Encoder()
    bm25.fit([doc.text for doc in documents])

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = str(Path(tmp_dir, "bm25_params.json"))
        bm25.dump(tmp_path)
        yield tmp_path


@pytest.fixture(scope="module")
def encoder(inner_encoder, bm_25_encoder_df_path):
    return HybridRecordEncoder(inner_encoder,
                               bm_25_encoder_df_path=bm_25_encoder_df_path,
                               batch_size=2)


def test_dimension(encoder, inner_dimension):
    assert encoder.dimension == inner_dimension


def test_init_encoder_invalid_alpha(inner_encoder):
    with pytest.raises(ValueError):
        HybridRecordEncoder(inner_encoder, alpha=-1)
    with pytest.raises(ValueError):
        HybridRecordEncoder(inner_encoder, alpha=2)
    with pytest.raises(ValueError):
        HybridRecordEncoder(inner_encoder, alpha=0)


def test_encode_documents(encoder, documents, queries):
    encoded_documents = encoder.encode_documents(documents)
    for encoded_document in encoded_documents:
        assert len(encoded_document.values) == encoder.dimension
        assert "indices" in encoded_document.sparse_values
        assert "values" in encoded_document.sparse_values


def test_encode_queries(encoder, queries):
    encoded_queries = encoder.encode_queries(queries)
    assert len(encoded_queries) == len(queries)
    for encoded_query in encoded_queries:
        assert len(encoded_query.values) == encoder.dimension
        assert "indices" in encoded_query.sparse_values
        assert "values" in encoded_query.sparse_values


def test_encode_queries_alpha_applied_correctly(encoder, queries):
    """
        Tests whether the alpha value is applied correctly when encoding queries.
    """
    alpha = 0.2
    alpha_coefficient = 2

    with patch.object(encoder, '_alpha', new=alpha):
        encoded_queries = encoder.encode_queries(queries)

    with patch.object(encoder, '_alpha', new=alpha_coefficient * alpha):
        encoded_queries_2 = encoder.encode_queries(queries)

    for encoded_query, encoded_query_2 in zip(encoded_queries, encoded_queries_2):
        assert len(encoded_query.values) == len(encoded_query_2.values)
        for value, value_2 in zip(encoded_query.values, encoded_query_2.values):
            assert pytest.approx(value * alpha_coefficient) == value_2


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
