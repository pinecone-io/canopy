import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pinecone_text.sparse import BM25Encoder

from canopy.knowledge_base.models import KBDocChunk
from canopy.knowledge_base.record_encoder import HybridRecordEncoder, DenseRecordEncoder
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
def dense_record_encoder(inner_dimension):
    return DenseRecordEncoder(StubDenseEncoder(dimension=inner_dimension))


@pytest.fixture(scope="module")
def bm_25_encoder_df_path(documents):
    bm25 = BM25Encoder()
    bm25.fit([doc.text for doc in documents])

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = str(Path(tmp_dir, "bm25_params.json"))
        bm25.dump(tmp_path)
        yield tmp_path


@pytest.fixture(scope="module")
def hybrid_encoder(dense_record_encoder, bm_25_encoder_df_path):
    return HybridRecordEncoder(dense_record_encoder,
                               bm_25_encoder_df_path=bm_25_encoder_df_path,
                               batch_size=2)


def test_dimension(hybrid_encoder, inner_dimension):
    assert hybrid_encoder.dimension == inner_dimension


def test_init_encoder_invalid_alpha(dense_record_encoder):
    with pytest.raises(ValueError):
        HybridRecordEncoder(dense_record_encoder, alpha=-1)
    with pytest.raises(ValueError):
        HybridRecordEncoder(dense_record_encoder, alpha=2)
    with pytest.raises(ValueError):
        HybridRecordEncoder(dense_record_encoder, alpha=0, match="sparse only")


def test_encode_documents(hybrid_encoder, documents, queries):
    encoded_documents = hybrid_encoder.encode_documents(documents)
    for encoded_document in encoded_documents:
        assert len(encoded_document.values) == hybrid_encoder.dimension
        assert "indices" in encoded_document.sparse_values
        assert "values" in encoded_document.sparse_values


def test_encode_queries(hybrid_encoder, queries):
    encoded_queries = hybrid_encoder.encode_queries(queries)
    assert len(encoded_queries) == len(queries)
    for encoded_query in encoded_queries:
        assert len(encoded_query.values) == hybrid_encoder.dimension
        assert "indices" in encoded_query.sparse_values
        assert "values" in encoded_query.sparse_values


def test_encode_queries_alpha_applied_correctly(dense_record_encoder,
                                                bm_25_encoder_df_path,
                                                queries):
    """
        Tests whether the alpha value is applied correctly when encoding queries.
    """
    alpha = 0.2
    alpha_coefficient = 2

    hb_1 = HybridRecordEncoder(dense_record_encoder,
                               bm_25_encoder_df_path=bm_25_encoder_df_path,
                               alpha=alpha)
    hb_2 = HybridRecordEncoder(dense_record_encoder,
                               bm_25_encoder_df_path=bm_25_encoder_df_path,
                               alpha=alpha_coefficient * alpha)

    encoded_queries = hb_1.encode_queries(queries)
    encoded_queries_2 = hb_2.encode_queries(queries)

    for encoded_query, encoded_query_2 in zip(encoded_queries, encoded_queries_2):
        assert len(encoded_query.values) == len(encoded_query_2.values)
        for value, value_2 in zip(encoded_query.values, encoded_query_2.values):
            assert pytest.approx(value * alpha_coefficient) == value_2

        assert (encoded_query.sparse_values["indices"] ==
                encoded_query_2.sparse_values["indices"])

        scaling_coefficient = (1 - alpha_coefficient * alpha) / (1 - alpha)
        for value, value_2 in zip(encoded_query.sparse_values["values"],
                                  encoded_query_2.sparse_values["values"]):

            assert pytest.approx(value * scaling_coefficient) == value_2


def test_encode_queries_with_alpha_1(hybrid_encoder, dense_record_encoder, queries):
    """
        Tests whether the encoded queries are exactly the same as the dense
        encoded queries when alpha is 1.
    """
    with patch.object(hybrid_encoder, '_alpha', new=1.0):
        encoded_queries = hybrid_encoder.encode_queries(queries)
        dense_queries = dense_record_encoder.encode_queries(queries)

        assert len(encoded_queries) == len(dense_queries)
        for encoded_query, dense_query in zip(encoded_queries, dense_queries):
            assert encoded_query.values == dense_query.values
