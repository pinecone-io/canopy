import pytest

from context_engine.knoweldge_base.document_encoder import DenseDocumentEncoder
from context_engine.knoweldge_base.models import KBDocChunk
from context_engine.models.data_models import Query
from ..stubs.stub_dense_encoder import StubDenseEncoder


class TestDenseDocumentEncoder:

    @pytest.fixture
    def dense_encoder(self):
        return StubDenseEncoder(dimension=3)

    @pytest.fixture
    def document_encoder(self, dense_encoder):
        return DenseDocumentEncoder(dense_encoder)

    @pytest.fixture
    def sample_docs(self):
        return [KBDocChunk(id=f"doc_1_{i}",
                           text=f"Sample document {i}",
                           document_id=f"doc_{i}",
                           metadata={"test": i})
                for i in range(5)]

    @pytest.fixture
    def sample_queries(self):
        return [Query(text="Sample query 1"), Query(text="Sample query 2")]

    def test_dimension(self, document_encoder, dense_encoder):
        assert document_encoder.dense_dimension == dense_encoder.dimension

    def test_encode_documents(self, document_encoder, dense_encoder, sample_docs):
        encoded_docs = document_encoder.encode_documents(sample_docs)
        assert len(encoded_docs) == len(sample_docs)
        for doc, sample_doc in zip(encoded_docs, sample_docs):
            assert len(doc.values) == document_encoder.dense_dimension
            assert doc.metadata == sample_doc.metadata
            assert doc.document_id == sample_doc.document_id
            assert doc.id == sample_doc.id
            assert doc.text == sample_doc.text
            expected_values = dense_encoder.encode_documents(sample_doc.text)
            assert doc.values == expected_values

    def test_encode_queries(self, document_encoder, sample_queries):
        encoded_queries = document_encoder.encode_queries(sample_queries)
        assert len(encoded_queries) == len(sample_queries)
        for q_encoded, q in zip(encoded_queries, sample_queries):
            assert len(q_encoded.values) == document_encoder.dense_dimension
            expected_values = document_encoder._dense_encoder.encode_queries(q.text)
            assert q_encoded.values == expected_values
            assert q_encoded.text == q.text

    def test_encode_empty_documents(self, document_encoder):
        encoded_docs = document_encoder.encode_documents([])
        assert encoded_docs == []

    def test_encode_empty_queries(self, document_encoder):
        encoded_queries = document_encoder.encode_queries([])
        assert encoded_queries == []

    @pytest.mark.asyncio
    async def test_aencode_documents_raises_not_implemented(self,
                                                            document_encoder,
                                                            sample_docs):
        with pytest.raises(NotImplementedError):
            await document_encoder.aencode_documents(sample_docs)

    @pytest.mark.asyncio
    async def test_aencode_queries_raises_not_implemented(self,
                                                          document_encoder,
                                                          sample_queries):
        with pytest.raises(NotImplementedError):
            await document_encoder.aencode_queries(sample_queries)
