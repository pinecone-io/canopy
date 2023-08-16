from ..stubs.stub_chunker import StubChunker
from context_engine.models.data_models import Document


class TestBaseChunker:

    @classmethod
    def setup_class(cls):
        cls.chunker = StubChunker()
        cls.documents = [Document(id="test_document_1",
                                  text="I am a simple test string to check the happy path of this simple chunker",
                                  metadata={"test": 1}),
                         Document(id="test_document_2",
                                  text="I am a simple test string to check the happy path of this simple chunker",
                                  metadata={"test": 2})]

    def test_chunker_chunk_documents(self):
        chunks = self.chunker.chunk_documents(self.documents)
        expected_chunks = [chunk for doc in self.documents for chunk in self.chunker.chunk_single_document(doc)]

        assert len(chunks) == len(expected_chunks)
        for chunk, expected_chunk in zip(chunks, expected_chunks):
            assert chunk.id == expected_chunk.id
            assert chunk.document_id == expected_chunk.document_id
            assert chunk.text == expected_chunk.text
            assert chunk.metadata == expected_chunk.metadata

        assert self.chunker.chunk_documents([]) == []
