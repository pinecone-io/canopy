import pytest
from context_engine.knoweldge_base.chunker.recursive_character \
    import RecursiveCharacterChunker
from context_engine.knoweldge_base.models import KBDocChunk
from tests.unit.chunker.base_test_chunker import BaseTestChunker
from tests.unit.stubs.stub_tokenizer import StubTokenizer


class TestRecursiveCharacterChunker(BaseTestChunker):

    @staticmethod
    @pytest.fixture(scope="class")
    def chunker():
        return RecursiveCharacterChunker(chunk_size=3,
                                         chunk_overlap=1,
                                         tokenizer=StubTokenizer())

    @staticmethod
    @pytest.fixture
    def expected_chunks(documents):
        return [
            KBDocChunk(id='test_document_1_0',
                       text='I am a',
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_1_1',
                       text='a simple test',
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_1_2',
                       text='test string to',
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_1_3',
                       text='to check the',
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_1_4',
                       text='the happy path',
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_1_5',
                       text='path of this',
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_1_6',
                       text='this simple chunker',
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_2_0',
                       text='another simple test',
                       metadata={'test': '2'},
                       document_id='test_document_2'),
            KBDocChunk(id='test_document_2_1',
                       text='test string',
                       metadata={'test': '2'},
                       document_id='test_document_2'),
            KBDocChunk(id='test_document_3_0',
                       text='sho',
                       metadata={'test': '2'},
                       document_id='test_document_3'),
            KBDocChunk(id='test_document_3_1',
                       text='ort',
                       metadata={'test': '2'},
                       document_id='test_document_3')]
