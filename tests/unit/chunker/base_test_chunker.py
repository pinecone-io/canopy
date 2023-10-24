import pytest
from abc import ABC, abstractmethod
from canopy.models.data_models import Document


class BaseTestChunker(ABC):

    @staticmethod
    @pytest.fixture(scope="class")
    @abstractmethod
    def chunker():
        pass

    @staticmethod
    @pytest.fixture
    def documents():
        return [
            Document(
                id="test_document_1",
                text="I am a simple test string"
                     " to check the happy path of this simple chunker",
                metadata={"test": 1}),
            Document(
                id="test_document_2",
                text="another simple test string",
                metadata={"test": 2},
                source="doc_2"
            ),
            Document(
                id="test_document_3",
                text="short",
                metadata={"test": 2},
                source="doc_3"
            )
        ]

    @staticmethod
    @pytest.fixture
    @abstractmethod
    def expected_chunks(documents):
        pass

    # region: test chunk_single_document

    @staticmethod
    def test_chunk_single_document_happy_path(chunker, documents, expected_chunks):
        for doc in documents:
            expected_chunks_for_doc = [chunk for chunk in
                                       expected_chunks if chunk.document_id == doc.id]
            actual_chunks = chunker.chunk_single_document(doc)
            assert len(actual_chunks) == len(expected_chunks_for_doc)
            for actual_chunk, expected_chunk in zip(actual_chunks,
                                                    expected_chunks_for_doc):
                assert actual_chunk == expected_chunk, f"actual: {actual_chunk}\n, " \
                                                       f"expected: {expected_chunk}"

    @staticmethod
    def test_chunk_single_document_empty_content(chunker, documents):
        empty_document = Document(id="test_document_3", text="", metadata={"test": 3})
        assert chunker.chunk_single_document(empty_document) == []

    # endregion

    # region: test achunk_single_document

    @staticmethod
    @pytest.mark.asyncio
    async def test_achunk_single_document_raise_error(chunker,
                                                      documents,
                                                      expected_chunks):
        with pytest.raises(NotImplementedError):
            await chunker.achunk_single_document(documents[0])

    # endregion

    # region: test chunk_documents

    @staticmethod
    def test_chunk_documents_happy_path(chunker,
                                        documents,
                                        expected_chunks):
        chunks = chunker.chunk_documents(documents)
        assert len(chunks) == len(expected_chunks)
        for chunk, expected_chunk in zip(chunks, expected_chunks):
            assert chunk == expected_chunk

    @staticmethod
    def test_chunk_documents_empty_list(chunker):
        assert chunker.chunk_documents([]) == []

    @staticmethod
    def test_chunk_documents_empty_content(chunker):
        empty_document = Document(id="test_document_3", text="", metadata={"test": 3})
        assert chunker.chunk_documents([empty_document]) == []

    # endregion

    # region: test achunk_documents

    @staticmethod
    @pytest.mark.asyncio
    async def test_achunk_documents_raise_error(chunker, documents):
        with pytest.raises(NotImplementedError):
            await chunker.achunk_documents(documents)

    # endregion
