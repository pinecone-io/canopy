from abc import ABC, abstractmethod
from typing import List

from resin.knoweldge_base.models import KBDocChunk
from resin.models.data_models import Document
from resin.utils.utils import FactoryMixin


class Chunker(ABC, FactoryMixin):

    """
    BaseChunker is an abstract class that defines the interface for a chunker.
    """

    def chunk_documents(self, documents: List[Document]) -> List[KBDocChunk]:
        chunks: List[KBDocChunk] = []
        for doc in documents:
            chunks.extend(self.chunk_single_document(doc))
        return chunks

    async def achunk_documents(self, documents: List[Document]) -> List[KBDocChunk]:
        chunks: List[KBDocChunk] = []
        for doc in documents:
            chunks.extend(await self.achunk_single_document(doc))
        return chunks

    @abstractmethod
    def chunk_single_document(self, document: Document) -> List[KBDocChunk]:
        pass

    @abstractmethod
    async def achunk_single_document(self, document: Document) -> List[KBDocChunk]:
        raise NotImplementedError()
