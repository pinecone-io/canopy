from abc import ABC, abstractmethod
from typing import List

from context_engine.knoweldge_base.models import KBEncodedDocChunk, KBQuery
from context_engine.models.data_models import PineconeDocumentRecord, PineconeQueryRecord

class Encoder(ABC):

    @abstractmethod
    def encode_documents(self, records: List[PineconeDocumentRecord]) -> List[PineconeDocumentRecord]:
        pass

    @abstractmethod
    def encode_queries(self, records: List[PineconeQueryRecord]) -> List[PineconeQueryRecord]:
        pass

    @abstractmethod
    async def aencode_documents(self, records: List[KBEncodedDocChunk]):
        pass

    @abstractmethod
    async def aencode_queries(self, records: List[KBQuery]):
        pass
