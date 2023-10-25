from typing import List

from canopy.knowledge_base.record_encoder import RecordEncoder
from canopy.knowledge_base.models import KBQuery, KBDocChunk, KBEncodedDocChunk
from canopy.models.data_models import Query
from .stub_dense_encoder import StubDenseEncoder


class StubRecordEncoder(RecordEncoder):

    def __init__(self,
                 stub_dense_encoder: StubDenseEncoder,
                 batch_size: int = 1):
        super().__init__(batch_size)
        self._dense_encoder = stub_dense_encoder

    def _encode_documents_batch(self,
                                documents: List[KBDocChunk]
                                ) -> List[KBEncodedDocChunk]:
        result: List[KBEncodedDocChunk] = []
        for doc in documents:
            values = self._dense_encoder.encode_documents(doc.text)
            result.append(
                KBEncodedDocChunk(
                    **doc.dict(),
                    values=values))
        return result

    def _encode_queries_batch(self,
                              queries: List[Query]
                              ) -> List[KBQuery]:
        result: List[KBQuery] = []
        for query in queries:
            values = self._dense_encoder.encode_queries(query.text)
            result.append(
                KBQuery(**query.dict(),
                        values=values))
        return result

    async def _aencode_documents_batch(self,
                                       documents: List[KBDocChunk]
                                       ) -> List[KBEncodedDocChunk]:
        raise NotImplementedError()

    async def _aencode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        raise NotImplementedError()

    @property
    def dimension(self) -> int:
        return self._dense_encoder.dimension
