from typing import List
from functools import cached_property
from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder

from .base import RecordEncoder
from resin.knowledge_base.models import KBQuery, KBEncodedDocChunk, KBDocChunk
from resin.models.data_models import Query


class DenseRecordEncoder(RecordEncoder):

    def __init__(self,
                 dense_encoder: BaseDenseEncoder,
                 **kwargs):
        super().__init__(**kwargs)
        self._dense_encoder = dense_encoder

    def _encode_documents_batch(self,
                                documents: List[KBDocChunk]
                                ) -> List[KBEncodedDocChunk]:
        dense_values = self._dense_encoder.encode_documents([d.text for d in documents])
        return [KBEncodedDocChunk(**d.dict(), values=v) for d, v in
                zip(documents, dense_values)]

    def _encode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        dense_values = self._dense_encoder.encode_queries([q.text for q in queries])
        return [KBQuery(**q.dict(), values=v) for q, v in zip(queries, dense_values)]

    @cached_property
    def dimension(self) -> int:
        return len(self._dense_encoder.encode_documents(["hello"])[0])

    async def _aencode_documents_batch(self,
                                       documents: List[KBDocChunk]
                                       ) -> List[KBEncodedDocChunk]:
        raise NotImplementedError

    async def _aencode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        raise NotImplementedError
