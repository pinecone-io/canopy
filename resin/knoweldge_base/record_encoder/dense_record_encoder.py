from typing import List, Optional
from functools import cached_property

from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder
from pinecone_text.dense.openai_encoder import OpenAIEncoder

from .base import RecordEncoder
from resin.knoweldge_base.models import KBQuery, KBEncodedDocChunk, KBDocChunk
from resin.models.data_models import Query


class DenseRecordEncoder(RecordEncoder):

    DEFAULT_DENSE_ENCODER = OpenAIEncoder
    DEFAULT_MODEL_NAME = "text-embedding-ada-002"

    def __init__(self,
                 dense_encoder: Optional[BaseDenseEncoder] = None,
                 *,
                 batch_size: int = 500,
                 **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)
        if dense_encoder is None:
            dense_encoder = self.DEFAULT_DENSE_ENCODER(self.DEFAULT_MODEL_NAME)
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
