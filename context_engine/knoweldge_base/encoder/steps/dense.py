from typing import List

from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder

from context_engine.knoweldge_base.encoder.steps.base import EncodingStep
from context_engine.knoweldge_base.models import KBQuery, KBEncodedDocChunk


class DenseEncodingStep(EncodingStep):

    def __init__(self, dense_encoder: BaseDenseEncoder):
        self.dense_encoder = dense_encoder

    def encode_documents(self, documents: List[KBEncodedDocChunk]):
        pass

    def encode_queries(self, queries: List[KBQuery]):
        for query in queries:
            query.values = self.dense_encoder.encode_query(query.text)

    async def aencode_documents(self, documents: List[KBEncodedDocChunk]):
        pass

    async def aencode_queries(self, queries: List[KBQuery]):
        pass