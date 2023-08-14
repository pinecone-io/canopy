from typing import List

from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder

from context_engine.knoweldge_base.encoder.base import Encoder
from context_engine.knoweldge_base.models import KBQuery, KBEncodedDocChunk


class DenseEncodingStep(Encoder):

    def __init__(self, dense_encoder: BaseDenseEncoder):
        self.dense_encoder = dense_encoder
        
    def _encode_documents_batch(self, documents: List[KBEncodedDocChunk]):
        pass

    # Alters the queries in place
    def _encode_queries_batch(self, queries: List[KBQuery]):
        pass

    async def _aencode_documents_batch(self, documents: List[KBEncodedDocChunk]):
        pass

    # Alters the queries in place
    async def _aencode_queries_batch(self, queries: List[KBQuery]):
        pass
