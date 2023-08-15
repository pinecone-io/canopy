from typing import List

from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder

from context_engine.knoweldge_base.encoder.steps.base import Encoder
from context_engine.knoweldge_base.models import KBQuery

from context_engine.knoweldge_base.encoder import EncodingPipeline 
from context_engine.knoweldge_base.encoder.dense import DenseEncoder



class HybridConvexScaler(Encoder):
    
        def __init__(self, alpha: float = 0.5):
            self.alpha = alpha

        def encode_documents(self, records: List[PineconeDocumentRecord]):
            return records
            

        def encode_queries(self, records: List[PineconeQueryRecord]):
            for query in queries:
                if query.values is None:
                    raise ValueError("Must run DenseEncodingStep before HybridEncodingStep")
    
                query.vector, query.sparse_vector = hybrid_convex_scale(query.values, query.sparse_values, self.alpha)


hybrid_encoder = EnodingPipeline([DenseEncoder("openai/ada2"), SparseEncoder("bm25"), HybridConvexScaler(alpha=0.7)])


class HybridEncoder(EncodingPipeline):
    
        def __init__(self, dense_model_name, sparse_model_name, alpha: float = 0.5):
            super().__init__([DenseEncoder(dense_model_name), SparseEncoder(sparse_model_name), HybridConvexScaler(alpha=alpha)])
            self.alpha = alpha

    
# hybrid_encoder = HybridEncoder("openai/ada2", "bm25", alpha=0.7)