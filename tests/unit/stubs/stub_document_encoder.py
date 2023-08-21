from typing import List

from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder

from context_engine.knoweldge_base.document_encoder.base_document_encoder \
    import BaseDocumentEncoder
from context_engine.knoweldge_base.models import KBQuery, KBDocChunk, KBEncodedDocChunk
from context_engine.models.data_models import Query
from .stub_dense_encoder import StubDenseEncoder


class StubDocumentEncoder(BaseDocumentEncoder):

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
            result.append(
                KBEncodedDocChunk(**doc.dict(),
                                  values=self._dense_encoder.encode_documents(doc.text)))
        return result

    def _encode_queries_batch(self,
                              queries: List[Query]
                              ) -> List[KBQuery]:
        result: List[KBQuery] = []
        for query in queries:
            result.append(
                KBQuery(**query.dict(),
                        values=self._dense_encoder.encode_queries(query.text)))
        return result

    async def _aencode_documents_batch(self,
                                       documents: List[KBDocChunk]
                                       ) -> List[KBEncodedDocChunk]:
        raise NotImplementedError()

    async def _aencode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        raise NotImplementedError()

    @property
    def dense_dimension(self) -> int:
        return self._dense_encoder.dimension
