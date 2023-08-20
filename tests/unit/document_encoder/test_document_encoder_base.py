import pytest
from context_engine.knoweldge_base.models import KBEncodedDocChunk, KBDocChunk
from context_engine.models.data_models import Query
from ..stubs.stub_document_encoder import StubDocumentEncoder


class TestDocumentEncoder:

    @pytest.fixture
    def encoder(self):
        return StubDocumentEncoder(batch_size=2, dimension=3)

    @pytest.fixture
    def sample_docs(self):
        return [KBDocChunk(id=f"doc_1_{i}",
                           text=f"Sample document {i}",
                           document_id=f"doc_{i}",
                           metadata={"test": i}) for i in range(5)]

    @pytest.fixture
    def sample_queries(self):
        return [Query(text="Sample query 1"), Query(text="Sample query 2")]

    def test_encode_documents(self, encoder, sample_docs):
        encoded_docs = encoder.encode_documents(sample_docs)
        assert len(encoded_docs) == len(sample_docs)
        for doc in encoded_docs:
            assert isinstance(doc, KBEncodedDocChunk)
            assert len(doc.values) == encoder.dimension

    def test_encode_queries(self, encoder, sample_queries):
        kb_queries = encoder.encode_queries(sample_queries)
        assert len(kb_queries) == len(sample_queries)
        for q in kb_queries:
            assert q.values
            assert len(q.values) == encoder.dimension

    def test_encode_documents_batch_size(self, encoder, sample_docs, mocker):
        mock_encode = mocker.patch.object(
            encoder, '_encode_documents_batch', wraps=encoder._encode_documents_batch)
        encoder.encode_documents(sample_docs)

        expected_call_count = -(-len(sample_docs) // encoder.batch_size)
        assert mock_encode.call_count == expected_call_count

        for idx, call in enumerate(mock_encode.call_args_list):
            args, _ = call
            batch = args[0]
            if idx < expected_call_count - 1:
                assert len(batch) == encoder.batch_size
            else:
                assert len(batch) == len(
                    sample_docs) % encoder.batch_size or encoder.batch_size

    @pytest.mark.asyncio
    async def test_aencode_documents_raises_not_implemented(self, encoder, sample_docs):
        with pytest.raises(NotImplementedError):
            await encoder.aencode_documents(sample_docs)

    @pytest.mark.asyncio
    async def test_aencode_queries_raises_not_implemented(self,
                                                          encoder,
                                                          sample_queries):
        with pytest.raises(NotImplementedError):
            await encoder.aencode_queries(sample_queries)

    def test_dense_dimension(self, encoder):
        assert encoder.dense_dimension == encoder.dimension
