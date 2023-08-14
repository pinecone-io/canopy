from abc import ABC, abstractmethod
from typing import List

from context_engine.knoweldge_base.models import KBEncodedDocChunk, KBQuery


class EncodingStep(ABC):

    @abstractmethod
    def encode_documents(self, documents: List[KBEncodedDocChunk]):
        pass

    @abstractmethod
    def encode_queries(self, queries: List[KBQuery]):
        pass

    @abstractmethod
    async def aencode_documents(self, documents: List[KBEncodedDocChunk]):
        pass

    @abstractmethod
    async def aencode_queries(self, queries: List[KBQuery]):
        pass
