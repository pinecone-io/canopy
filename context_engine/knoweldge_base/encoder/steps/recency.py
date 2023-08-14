from typing import List

from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder

from context_engine.knoweldge_base.encoder.steps.base import EncodingStep
from context_engine.knoweldge_base.models import KBQuery


class RecencyEncodingStep(EncodingStep):
    def __init__(self,
                 recency_encoder: BaseSparseEncoder, default_beta: float = 0.5):
        self.recency_encoder = recency_encoder
        self.default_beta = default_beta

    def encode_queries(self, queries: List[KBQuery]):
        for query in queries:
            beta = query.query_params.get("beta", self.default_beta)
            recency_values = self.recency_encoder.encode_query(query.text)
            # TODO: implement recency scaling
            query.values, query.sparse_values = recency_scaling(query.values,
                                                                query.sparse_values,
                                                                recency_values,
                                                                beta)
