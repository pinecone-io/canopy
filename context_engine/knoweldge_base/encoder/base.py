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

    @abstractmethod
    def _encode_documents_batch(self, documents: List[KBDocChunk]) -> List[KBEncodedDocChunk]:
        pass

    @abstractmethod
    def _encode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        pass

    @abstractmethod
    async def _aencode_documents_batch(self, documents: List[KBDocChunk]) -> List[KBEncodedDocChunk]:
        raise NotImplementedError

    @abstractmethod
    async def _aencode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        raise NotImplementedError

    @staticmethod
    def _batch_iterator(data: list, batch_size):
        return (data[pos:pos + batch_size] for pos in range(0, len(data), batch_size))

    def encode_documents(self, documents: List[KBDocChunk]) -> List[KBEncodedDocChunk]:
        encoded_docs = []
        for batch in self._batch_iterator(documents, self.batch_size):
            encoded_docs.append(self._encode_documents_batch(batch))

        return encoded_docs

    def encode_queries(self, queries: List[Query]) -> List[KBQuery]:
        kb_queries = []
        for batch in self._batch_iterator(queries, self.batch_size):
            kb_queries.append(self._encode_queries_batch(batch))

        return kb_queries

    async def aencode_documents(self, documents: List[KBDocChunk]) -> List[KBEncodedDocChunk]:
        encoded_docs = []
        for batch in self._batch_iterator(documents, self.batch_size):
            encoded_docs.append(await self._aencode_documents_batch(batch))

        return encoded_docs


    async def aencode_queries(self, queries: List[Query]) -> List[KBQuery]:
        kb_queries = []
        for batch in self._batch_iterator(queries, self.batch_size):
            kb_queries.append(await self._aencode_queries_batch(batch))

        return kb_queries

