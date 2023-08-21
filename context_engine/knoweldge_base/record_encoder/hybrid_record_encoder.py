from typing import List

from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder
from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder

from context_engine.knoweldge_base.record_encoder.base_record_encoder import\
    BaseRecordEncoder
from context_engine.knoweldge_base.models import KBQuery, KBEncodedDocChunk, KBDocChunk
from context_engine.models.data_models import Query


# TODO: decide if we want to remove before release

class HybridRecordEncoder(BaseRecordEncoder):
    """
    A hybrid encoder that combines dense embeddings and a sparse representation
    (e.g. keyword counts) using a linear  combination of the two.
    The alpha value for the convex combination can be specified in the query_params
    """

    def __init__(self,
                 dense_encoder: BaseDenseEncoder,
                 sparse_encoder: BaseSparseEncoder,
                 default_alpha: float = 0.5,
                 **kwargs
                 ):
        """

        Args:
            dense_encoder: A DenseEncoder from pinecone_text that will be used to
            generate dense embeddings
            sparse_encoder: A SparseEncoder from pinecone_text that will be used to
                            generate sparse representations
            default_alpha: The default alpha value to use for scaling dense and
                           sparse vectors. Defaults to 0.5.
                           Each query may include its own alpha value in the
                           query_params dict, which will override this value.

        Keyword Args:
            batch_size: The number of documents or queries to encode at once.
            Defaults to 1.
        """
        super().__init__(**kwargs)
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.default_alpha = default_alpha

    def _encode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        values = self.dense_encoder.encode_query([q.text for q in queries])
        sparse_values = self.sparse_encoder.encode_query([q.text for q in queries])

        encoded_queries = []
        for query, val, sparse_val in zip(queries, values, sparse_values):
            if query.query_params and "alpha" in query.query_params:
                alpha = query.query_params["alpha"]
            else:
                alpha = self.default_alpha
            values, sparse_values = hybrid_convex_scale(val, sparse_val, alpha)
            encoded_queries.append(
                KBQuery(**query.dict(), values=values, sparse_values=sparse_values)
            )
        return encoded_queries

    def _encode_documents_batch(self,
                                documents: List[KBDocChunk]
                                ) -> List[KBEncodedDocChunk]:
        raise NotImplementedError

    async def _aencode_documents_batch(self,
                                       documents: List[KBDocChunk]
                                       ) -> List[KBEncodedDocChunk]:
        raise NotImplementedError

    async def _aencode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        raise NotImplementedError
