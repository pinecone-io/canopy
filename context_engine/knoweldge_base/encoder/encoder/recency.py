from typing import List

from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder

from context_engine.knoweldge_base.encoder import Encoder
from context_engine.knoweldge_base.models import KBQuery


class RecencyEncoder(Encoder):
    def __init__(self, beta: float = 0.5):
        # TODO: implement
        pass

    def encode_documents(self, records: List[PineconeDocumentRecord]):
        # TODO: do something
        pass

    def encode_queries(self, records: List[PineconeQueryRecord]):
        for query in queries:
            beta = query.query_params.get("beta", self.default_beta)
            recency_values = self.recency_encoder.encode_query(query.text)
            # TODO: implement recency scaling
            query.values, query.sparse_values = recency_scaling(query.values,
                                                                query.sparse_values,
                                                                recency_values,
                                                                beta)
