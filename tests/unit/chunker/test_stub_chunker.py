import pytest

from context_engine.knoweldge_base.models import KBDocChunk
from .base_test_chunker import BaseTestChunker
from ..stubs.stub_chunker import StubChunker


class TestStubChunker(BaseTestChunker):

    @staticmethod
    @pytest.fixture(scope="class")
    def chunker():
        return StubChunker()

    @staticmethod
    @pytest.fixture
    def expected_chunks(documents):
        return [KBDocChunk(id=f"{document.id}_0",
                           document_id=document.id,
                           text=document.text,
                           metadata=document.metadata)
                for document in documents]
