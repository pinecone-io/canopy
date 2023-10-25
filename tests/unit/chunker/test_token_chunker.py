import pytest

from canopy.knowledge_base.models import KBDocChunk
from canopy.models.data_models import Document
from .base_test_chunker import BaseTestChunker
from canopy.knowledge_base.chunker.token_chunker import TokenChunker


class TestTokenChunker(BaseTestChunker):

    @staticmethod
    @pytest.fixture(scope="class")
    def chunker():
        return TokenChunker(max_chunk_size=5,
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
                           document_id='test_document_1',),
                KBDocChunk(id='test_document_2_0',
                           text='another simple test string',
                           metadata={'test': '2'},
                           document_id='test_document_2',
                           source='doc_2'),
                KBDocChunk(id='test_document_3_0',
                           text='short',
                           metadata={'test': '2'},
                           document_id='test_document_3',
                           source='doc_3'),
                ]

    @staticmethod
    def test_chunk_single_document_zero_overlap(chunker):
        chunker._overlap = 0
        document = Document(id="test_document_1",
                            text="I am a test string with no overlap",
                            metadata={"test": 1})
        actual = chunker.chunk_single_document(document)

        expected = [KBDocChunk(id='test_document_1_0',
                               text='I am a test string',
                               metadata={'test': '1'},
                               document_id='test_document_1'),
                    KBDocChunk(id='test_document_1_1',
                               text='with no overlap',
                               metadata={'test': '1'},
                               document_id='test_document_1')]

        for actual_chunk, expected_chunk in zip(actual, expected):
            assert actual_chunk == expected_chunk

    @staticmethod
    def test_chunker_init_raise_on_negative_overlap():
        with pytest.raises(ValueError):
            TokenChunker(max_chunk_size=5,
                         overlap=-1)

    @staticmethod
    def test_chunker_init_raise_on_non_positive_max_tokens():
        with pytest.raises(ValueError):
            TokenChunker(max_chunk_size=0,
                         overlap=5)
