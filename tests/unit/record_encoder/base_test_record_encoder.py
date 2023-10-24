import pytest
import math
from abc import ABC, abstractmethod

from canopy.knowledge_base.models import KBDocChunk, KBEncodedDocChunk, KBQuery
from canopy.models.data_models import Query


class BaseTestRecordEncoder(ABC):

    @staticmethod
    @pytest.fixture
    @abstractmethod
    def expected_dimension():
        pass

    @staticmethod
    @pytest.fixture
    @abstractmethod
    def inner_encoder(expected_dimension):
        pass

    @staticmethod
    @pytest.fixture
    @abstractmethod
    def record_encoder(inner_encoder):
        pass

    @staticmethod
    @pytest.fixture
    def documents():
        return [KBDocChunk(id=f"doc_1_{i}",
                           text=f"Sample document {i}",
                           document_id=f"doc_{i}",
                           metadata={"test": i},
                           source="doc_1")
                for i in range(5)]

    @staticmethod
    @pytest.fixture
    def queries():
        return [Query(text="Sample query 1"),
                Query(text="Sample query 2"),
                Query(text="Sample query 3")]

    @staticmethod
    @pytest.fixture
    def expected_encoded_documents(documents, inner_encoder):
        values = inner_encoder.encode_documents([d.text for d in documents])
        return [KBEncodedDocChunk(**d.dict(), values=v) for d, v in
                zip(documents, values)]

    @staticmethod
    @pytest.fixture
    def expected_encoded_queries(queries, inner_encoder):
        values = inner_encoder.encode_queries([q.text for q in queries])
        return [KBQuery(**q.dict(), values=v) for q, v in zip(queries, values)]

    @staticmethod
    def test_dimension(record_encoder, expected_dimension):
        assert record_encoder.dimension == expected_dimension

    # region: test encode_documents

    @staticmethod
    def test_encode_documents_one_by_one(record_encoder,
                                         documents,
                                         expected_encoded_documents,
                                         expected_dimension,
                                         mocker):
        record_encoder.batch_size = 1
        mock_encode = mocker.patch.object(
            record_encoder, '_encode_documents_batch',
            wraps=record_encoder._encode_documents_batch)

        actual = record_encoder.encode_documents(documents)

        assert mock_encode.call_count == len(expected_encoded_documents)
        assert actual == expected_encoded_documents

    @staticmethod
    def test_encode_documents_batches(documents,
                                      expected_encoded_documents,
                                      record_encoder,
                                      mocker):
        record_encoder.batch_size = 2
        mock_encode = mocker.patch.object(
            record_encoder, '_encode_documents_batch',
            wraps=record_encoder._encode_documents_batch)
        actual = record_encoder.encode_documents(documents)

        expected_call_count = math.ceil(len(documents) / record_encoder.batch_size)
        assert mock_encode.call_count == expected_call_count

        for idx, call in enumerate(mock_encode.call_args_list):
            args, _ = call
            batch = args[0]
            if idx < expected_call_count - 1:
                assert len(batch) == record_encoder.batch_size
            else:
                assert len(batch) == len(
                    documents) % record_encoder.batch_size or record_encoder.batch_size

        assert actual == expected_encoded_documents

    @staticmethod
    def test_encode_empty_documents_list(record_encoder):
        assert record_encoder.encode_documents([]) == []

    # endregion

    # region: test aencode_documents

    @staticmethod
    @pytest.mark.asyncio
    async def test_aencode_documents_not_implemented(record_encoder,
                                                     documents):
        with pytest.raises(NotImplementedError):
            await record_encoder.aencode_documents(documents)

    # endregion

    # region: test encode_queries

    @staticmethod
    def test_encode_queries_one_by_one(record_encoder,
                                       queries,
                                       expected_encoded_queries,
                                       mocker):
        record_encoder.batch_size = 1
        mock_encode = mocker.patch.object(record_encoder, '_encode_queries_batch',
                                          wraps=record_encoder._encode_queries_batch)
        actual = record_encoder.encode_queries(queries)

        assert mock_encode.call_count == len(expected_encoded_queries)
        assert actual == expected_encoded_queries

    @staticmethod
    def test_encode_queries_batches(queries,
                                    expected_encoded_queries,
                                    record_encoder, mocker):
        record_encoder.batch_size = 2
        mock_encode = mocker.patch.object(record_encoder, '_encode_queries_batch',
                                          wraps=record_encoder._encode_queries_batch)
        actual = record_encoder.encode_queries(queries)

        expected_call_count = math.ceil(len(queries) / record_encoder.batch_size)
        assert mock_encode.call_count == expected_call_count

        for idx, call in enumerate(mock_encode.call_args_list):
            args, _ = call
            batch = args[0]
            if idx < expected_call_count - 1:
                assert len(batch) == record_encoder.batch_size
            else:
                assert len(batch) == len(queries) % record_encoder.batch_size \
                       or record_encoder.batch_size

        assert actual == expected_encoded_queries

    @staticmethod
    def test_encode_empty_queries_list(record_encoder):
        assert record_encoder.encode_queries([]) == []

    # endregion

    # region: test aencode_queries

    @staticmethod
    @pytest.mark.asyncio
    async def test_aencode_queries_not_implemented(record_encoder, queries):
        with pytest.raises(NotImplementedError):
            await record_encoder.aencode_queries(queries)

    # endregion
