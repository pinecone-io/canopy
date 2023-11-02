from abc import ABC, abstractmethod
from typing import List

from canopy.knowledge_base.models import KBDocChunk
from canopy.models.data_models import Document
from canopy.utils.config import ConfigurableMixin


class Chunker(ABC, ConfigurableMixin):

    """
    Base class for chunkers. Chunkers take a document (id, text, ...)
    and return a list of KBDocChunks  (id, text, document_id, ...)
    Chunker is an abstract class that must be subclassed to be used,
    also, it extends ConfigurableMixin which means that every subclass of
    Chunker could be referenced by a name and configured in a config file.
    """

    def chunk_documents(self, documents: List[Document]) -> List[KBDocChunk]:
        """
        chunk_documents takes a list of documents and returns a list of KBDocChunks
        this method is just a wrapper around chunk_single_document that can be
        used to chunk a list of documents.

        Args:
            documents: list of documents

        Returns:
            chunks: list of chunks of type KBDocChunks
        """
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
        """
        chunk_single_document takes a document and returns a
        list of KBDocChunks, this is the main method
        that must be implemented by every subclass of Chunker

        Args:
            document: list of documents

        Returns:
            chunks: list of chunks KBDocChunks
        """
        pass

    @abstractmethod
    async def achunk_single_document(self, document: Document) -> List[KBDocChunk]:
        raise NotImplementedError()
