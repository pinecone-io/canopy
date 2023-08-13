from abc import ABC, abstractmethod
from typing import List

from context_engine.knoweldge_base.models import KBDocChunk, KBQuery, KBEncodedDocChunk
from context_engine.models.data_models import Query


class Encoder(ABC):

    # TODO: decided whether we want to return a new list, or edit the KBDocChunks in place
    # (for now, assuming we edit in place)
    @abstractmethod
    def encode_documents(self, documents: List[KBDocChunk]
                         ) -> List[KBEncodedDocChunk]:
        pass

    @abstractmethod
    def encode_queries(self, queries: List[Query]
                       ) -> List[KBQuery]:
        pass

    @abstractmethod
    async def aencode_documents(self, documents: List[KBDocChunk]
                                ) -> List[KBEncodedDocChunk]:
        pass

    @abstractmethod
    async def aencode_queries(self, queries: List[Query]
                              ) -> List[KBQuery]:
        pass
