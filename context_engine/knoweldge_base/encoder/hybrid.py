from typing import List

from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder
from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder

from context_engine.knoweldge_base.encoder.base import Encoder
from context_engine.knoweldge_base.models import KBQuery


class HybridEncoder(Encoder):

    def __init__(self,
                 dense_encoder: BaseDenseEncoder,
                 sparse_encoder: BaseSparseEncoder,
                 default_alpha: float = 0.5):
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.default_alpha = default_alpha

    def _encode_queries_batch(self, queries: List[KBQuery]):
        # NOTE: assumes that a DenseEncodingStep was run first
        values = self.dense_encoder.encode_query([q.text for q in queries])
        sparse_values = self.sparse_encoder.encode_query([q.text for q in queries])

        for query, val, sparse_val in zip(queries, values, sparse_values):
            alpha = query.query_params.get("alpha", self.default_alpha)
            query.values, query.sparse_values = hybrid_convex_scale(val, sparse_val, alpha)
