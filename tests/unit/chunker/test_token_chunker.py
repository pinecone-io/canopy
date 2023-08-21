import pytest

from context_engine.knoweldge_base.models import KBDocChunk
from .base_test_chunker import BaseTestChunker
from ..stubs.stub_tokenizer import StubTokenizer
from context_engine.knoweldge_base.chunker.token_chunker import TokenChunker


class TestTokenChunker(BaseTestChunker):

    @staticmethod
    @pytest.fixture(scope="class")
    def chunker():
        return TokenChunker(tokenizer=StubTokenizer(),
                            max_chunk_size=5,
                            overlap=2)

    @staticmethod
    @pytest.fixture
    def expected_chunks(documents):
        return [KBDocChunk(id='test_document_1_0',
                           text='I am a simple test',
                           metadata={'test': '1'},
                           document_id='test_document_1'),
                KBDocChunk(id='test_document_1_1',
                           text='simple test string to check',
                           metadata={'test': '1'},
                           document_id='test_document_1'),
                KBDocChunk(id='test_document_1_2',
                           text='to check the happy path',
                           metadata={'test': '1'},
                           document_id='test_document_1'),
                KBDocChunk(id='test_document_1_3',
                           text='happy path of this simple',
                           metadata={'test': '1'},
                           document_id='test_document_1'),
                KBDocChunk(id='test_document_1_4',
                           text='this simple chunker',
                           metadata={'test': '1'},
                           document_id='test_document_1'),
                KBDocChunk(id='test_document_2_0',
                           text='another simple test string',
                           metadata={'test': '2'},
                           document_id='test_document_2')]
