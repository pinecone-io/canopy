from abc import ABC, abstractmethod
from typing import List, Optional

from canopy.knowledge_base.models import KBEncodedDocChunk, KBQuery, KBDocChunk
from canopy.models.data_models import Query
from canopy.utils.config import ConfigurableMixin


class RecordEncoder(ABC, ConfigurableMixin):
    """
    Base class for all encoders. Encoders are used to encode documents' and queries'
    text into vectors.

        Args:
            batch_size: The number of documents or queries to encode at once.
            Defaults to 1.
    """

    def __init__(self, batch_size: int = 1):

        self.batch_size = batch_size

    @abstractmethod
    def _encode_documents_batch(self,
                                documents: List[KBDocChunk]
                                ) -> List[KBEncodedDocChunk]:
        pass

    @abstractmethod
    def _encode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
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
        encoded_docs = []
        for batch in self._batch_iterator(documents, self.batch_size):
            encoded_docs.extend(self._encode_documents_batch(batch))

        return encoded_docs

    def encode_queries(self, queries: List[Query]) -> List[KBQuery]:
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
