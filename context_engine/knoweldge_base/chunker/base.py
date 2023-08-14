from abc import ABC, abstractmethod
from typing import List

from context_engine.knoweldge_base.models import KBDocChunk
from context_engine.models.data_models import Document


class Chunker(ABC):
    """
    BaseChunker is an abstract class that defines the interface for a chunker.
    """

    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[KBDocChunk]:
        pass

    @abstractmethod
    async def achunk_documents(self, documents: List[Document]) -> List[KBDocChunk]:
        pass
