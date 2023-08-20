from ..stubs.stub_tokenizer import StubTokenizer
from context_engine.knoweldge_base.chunker.token_chunker import TokenChunker
from context_engine.models.data_models import Document


class TestTokenChunker:

    """
    Note: tests are minimal since we want
    to remove this chunker before first release
    """

    @classmethod
    def setup_class(cls):

        cls.chunker = TokenChunker(tokenizer=StubTokenizer(),
                                   max_chunk_size=5,
                                   overlap=2)
        cls.text = "I am a simple test string to check the happy path"
        cls.expected_texts = ["I am a simple test",
                              "simple test string to check",
                              "to check the happy path"]

    def test_chunk_single_document(self):
        document = Document(id="test_document", text=self.text, metadata={})
        chunks = self.chunker.chunk_single_document(document)
        assert len(chunks) == len(self.expected_texts)
        for chunk, expected_text in zip(chunks, self.expected_texts):
            assert chunk.text == expected_text
            assert chunk.document_id == document.id
            assert chunk.metadata == document.metadata
