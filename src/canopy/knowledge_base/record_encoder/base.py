from abc import ABC, abstractmethod
from typing import List, Optional

from canopy.knowledge_base.models import KBEncodedDocChunk, KBQuery, KBDocChunk
from canopy.models.data_models import Query
from canopy.utils.config import ConfigurableMixin


class RecordEncoder(ABC, ConfigurableMixin):
    """
    Base class for encoders. Encoders code text and to vector representations - dense and sparse.

    Canopy RecordEncoder implementation seperates the encoding of documents and queries: we do it since 
    some implementation of both sparse and dense encoding are not symmetrical. For example, BM25 sparse 
    encoders and instruction dense encoders.

    Additionally, the implementation of the encoding is per batch, so every class that extends RecordEncoder
    should implement the following methods at minumum:
    - _encode_documents_batch
    - _encode_queries_batch
    
    Async encoders are still not supported, but will be added in the future.

    Args:
        batch_size: The number of documents or queries to encode at once.
        Defaults to 1.
    
    """

    def __init__(self, batch_size: int = 1):
        """
        Initialize the encoder.

        Args:
            batch_size: The number of documents or queries to encode at once.
                        Defaults to 1.
        """
        self.batch_size = batch_size

    @abstractmethod
    def _encode_documents_batch(self,
                                documents: List[KBDocChunk] # TODO: rename documents to doc_chunks or chunks
                                ) -> List[KBEncodedDocChunk]:
        """
        
        Abstract method for encoding a batch of documents, takes a list of KBDocChunk and returns a list of KBEncodedDocChunk.
        The implementation of this method should be batched, meaning that it should encode the documents in batches of size

        Args:
            documents: A list of KBDocChunk to encode.

        Returns:
            encoded chunks: A list of KBEncodedDocChunk.
        
        """
        pass

    @abstractmethod
    def _encode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        """

        Abstract method for encoding a batch of queries, takes a list of Query and returns a list of KBQuery.
        The implementation of this method should be batched, meaning that it should encode the queries in batches of size

        Args:
            queries: A list of Query to encode.

        Returns:
            encoded queries: A list of KBQuery.
        """
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
        """
        return None

    def encode_documents(self, documents: List[KBDocChunk]) -> List[KBEncodedDocChunk]:
        """
        
        Encode documents in batches. Will iterate over batch of documents and encode them using the _encode_documents_batch method.

        Args:
            documents: A list of KBDocChunk to encode.

        Returns:
            encoded chunks: A list of KBEncodedDocChunk.

        """
        encoded_docs = []
        for batch in self._batch_iterator(documents, self.batch_size):
            encoded_docs.extend(self._encode_documents_batch(batch))

        return encoded_docs # TODO: consider yielding a generator

    def encode_queries(self, queries: List[Query]) -> List[KBQuery]:
        """

        Encode queries in batches. Will iterate over batch of queries and encode them using the _encode_queries_batch method.

        Args:
            queries: A list of Query to encode.
        
        Returns:
            encoded queries: A list of KBQuery.
        """
        
        kb_queries = []
        for batch in self._batch_iterator(queries, self.batch_size):
            kb_queries.extend(self._encode_queries_batch(batch))

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
