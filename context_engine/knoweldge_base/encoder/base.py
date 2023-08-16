from abc import ABC, abstractmethod
from typing import List

from context_engine.knoweldge_base.models import KBEncodedDocChunk, KBQuery, KBDocChunk
from context_engine.models.data_models import Query


class Encoder(ABC):
    """
    Base class for all encoders. Encoders are used to encode documents' and queries' text into vectors.
    """

    def __init__(self, batch_size: int = 1):
        """
        Args:
            batch_size: The number of documents or queries to encode at once. Defaults to 1.
        """
        self.batch_size = batch_size

    # Alters the documents in place
    @abstractmethod
    def _encode_documents_batch(self, documents: List[KBEncodedDocChunk]):
        pass

    # Alters the queries in place
    @abstractmethod
    def _encode_queries_batch(self, queries: List[KBQuery]):
        pass

    @abstractmethod
    async def _aencode_documents_batch(self, documents: List[KBEncodedDocChunk]):
        raise NotImplementedError

    # Alters the queries in place
    @abstractmethod
    async def _aencode_queries_batch(self, queries: List[KBQuery]):
        raise NotImplementedError

    @staticmethod
    def _batch_iterator(data: list, batch_size):
        return (data[pos:pos + batch_size] for pos in range(0, len(data), batch_size))

    def encode_documents(self, documents: List[KBDocChunk]) -> List[KBEncodedDocChunk]:
        encoded_chunks = [KBEncodedDocChunk(**doc.dict()) for doc in documents]
        for batch in self._batch_iterator(encoded_chunks, self.batch_size):
            self._encode_documents_batch(batch)

        return encoded_chunks

    def encode_queries(self, queries: List[Query]) -> List[KBQuery]:
        kb_queries = [KBQuery(**query.dict()) for query in queries]
        for batch in self._batch_iterator(kb_queries, self.batch_size):
            self._encode_queries_batch(batch)

        return kb_queries

    async def aencode_documents(self, documents: List[KBEncodedDocChunk]):
        encoded_chunks = [KBEncodedDocChunk(**doc.dict()) for doc in documents]
        for batch in self._batch_iterator(encoded_chunks, self.batch_size):
            await self._aencode_documents_batch(batch)

        return encoded_chunks


    async def aencode_queries(self, queries: List[KBQuery]):
        kb_queries = [KBQuery(**query.dict()) for query in queries]
        for batch in self._batch_iterator(kb_queries, self.batch_size):
            await self._aencode_queries_batch(batch)

        return kb_queries

