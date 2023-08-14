from typing import List

from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder

from context_engine.knoweldge_base.encoder.steps.base import Encoder
from context_engine.knoweldge_base.models import KBQuery


class HybridEncodingStep(Encoder):

    def __init__(self, sparse_encoder: BaseSparseEncoder, default_alpha: float = 0.5):
        self.sparse_encoder = sparse_encoder
        self.default_alpha = default_alpha

    def encode_queries(self, queries: List[KBQuery]):
        # NOTE: assumes that a DenseEncodingStep was run first
        for query in queries:
            if query.values is None:
                raise ValueError("Must run DenseEncodingStep before HybridEncodingStep")

            alpha = query.query_params.get("alpha", self.default_alpha)
            query.sparse_values = self.sparse_encoder.encode_query(query.text)
            query.values, query.sparse_values = hybrid_convex_scale(query.values, query.sparse_values, alpha)
