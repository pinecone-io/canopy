from abc import ABC, abstractmethod
from typing import List, Optional

from src.canopy.knowledge_base.models import KBEncodedDocChunk, KBQuery, KBDocChunk
from src.canopy.models.data_models import Query
from src.canopy.utils.config import ConfigurableMixin


class RecordEncoder(ABC, ConfigurableMixin):
    """
    Base class for RecordEncoders. Encodes document chunks and queries to vector representations.
    The vector representation may include both dense and sparse values.
    Dense values are usually generated by an embedding model, and sparse values usually represent weighted keyword counts.

    The RecordEncoder implements separate functions for the encoding of documents and queries.
    Some implementations of both sparse and dense encoding are not symmetrical. For example, BM25 sparse
    encoders and instruction dense encoders.

    Any class that extends RecordEncoder must implement the methods for encoding documents and queries:
    - _encode_documents_batch
    - _encode_queries_batch

    Async encoders are still not supported, but will be added in the future.
    """   # noqa: E501

    def __init__(self, batch_size: int = 1):
        """
        Initialize the encoder.

        Args:
            batch_size: The number of documents or queries to encode at once.
                        Defaults to 1.
        """   # noqa: E501
        self.batch_size = batch_size

    # TODO: rename documents to doc_chunks or chunks
    @abstractmethod
    def _encode_documents_batch(self,
                                documents: List[KBDocChunk]
                                ) -> List[KBEncodedDocChunk]:
        """
        Abstract method for encoding a batch of documents, takes a list of KBDocChunk and returns a list of KBEncodedDocChunk.
        For maximal performance, and derived class should try to operate on the entire documents batch in a single operation.

        Args:
            documents: A list of KBDocChunk to encode.

        Returns:
            encoded chunks: A list of KBEncodedDocChunk.
        """   # noqa: E501
        pass

    @abstractmethod
    def _encode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        """
        Abstract method for encoding a batch of queries, takes a list of Query and returns a list of KBQuery.
        For maximal performance, and derived class should try to operate on the entire batch in a single operation.

        Args:
            queries: A list of `Query` objects to encode.

        Returns:
            encoded queries: A list of KBQuery.
        """   # noqa: E501
        pass

    @abstractmethod
    async def _aencode_documents_batch(self,
                                       documents: List[KBDocChunk]
                                       ) -> List[KBEncodedDocChunk]:
        raise NotImplementedError

    @abstractmethod
    async def _aencode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        raise NotImplementedError

    @staticmethod
    def _batch_iterator(data: list, batch_size):
        return (data[pos:pos + batch_size] for pos in range(0, len(data), batch_size))

    @property
    def dimension(self) -> Optional[int]:
        """
        Returns:
            The dimension of the dense vectors produced by the encoder, if applicable.
        """   # noqa: E501
        return None

    def encode_documents(self, documents: List[KBDocChunk]) -> List[KBEncodedDocChunk]:
        """

        Encode documents in batches. Will iterate over batch of documents and encode them using the _encode_documents_batch method.

        Args:
            documents: A list of KBDocChunk to encode.

        Returns:
            encoded chunks: A list of KBEncodedDocChunk.

        """   # noqa: E501
        encoded_docs = []
        for batch in self._batch_iterator(documents, self.batch_size):
            try:
                encoded_docs.extend(self._encode_documents_batch(batch))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to enconde documents using {self.__class__.__name__}. "
                    f"Error: {self._format_error(e)}"
                ) from e

        return encoded_docs  # TODO: consider yielding a generator

    def encode_queries(self, queries: List[Query]) -> List[KBQuery]:
        """

        Encode queries in batches. Will iterate over batch of queries and encode them using the _encode_queries_batch method.

        Args:
            queries: A list of Query to encode.

        Returns:
            encoded queries: A list of KBQuery.
        """   # noqa: E501

        kb_queries = []
        for batch in self._batch_iterator(queries, self.batch_size):
            try:
                kb_queries.extend(self._encode_queries_batch(batch))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to enconde queries using {self.__class__.__name__}. "
                    f"Error: {self._format_error(e)}"
                ) from e

        return kb_queries

    async def aencode_documents(self,
                                documents: List[KBDocChunk]
                                ) -> List[KBEncodedDocChunk]:
        encoded_docs = []
        for batch in self._batch_iterator(documents, self.batch_size):
            encoded_docs.extend(await self._aencode_documents_batch(batch))

        return encoded_docs

    async def aencode_queries(self, queries: List[Query]) -> List[KBQuery]:
        kb_queries = []
        for batch in self._batch_iterator(queries, self.batch_size):
            kb_queries.extend(await self._aencode_queries_batch(batch))

        return kb_queries

    def _format_error(self, err):
        return f"{err}"
