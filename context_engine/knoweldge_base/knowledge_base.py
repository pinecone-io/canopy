from typing import List, Optional

from context_engine.knoweldge_base.base_knoweldge_base import BaseKnowledgeBase
from context_engine.knoweldge_base.chunker.base import Chunker
from context_engine.knoweldge_base.encoder.base import Encoder
from context_engine.knoweldge_base.models import KBQueryResult, KBQuery, QueryResult
from context_engine.knoweldge_base.reranker.reranker import Reranker, TransparentReranker
from context_engine.knoweldge_base.tokenizer.base import Tokenizer
from context_engine.models.data_models import Query


class KnowledgeBase(BaseKnowledgeBase):
    def __init__(self,
                 index_name: str,
                 *,
                 encoder: Encoder,
                 tokenizer: Tokenizer,
                 chunker: Chunker,
                 reranker: Optional[Reranker] = None,
                 default_top_k: int = 10,
                 indexed_fields: List[str] = ['document_id'],
                 ):

        self.index_name = index_name

        # TODO: decide how we are instantiating the encoder - as a single encoder that does both dense and
        #       sparse or as two separate encoders
        self._encoder = encoder
        self._tokenizer = tokenizer
        self._chunker = chunker
        self._reranker = TransparentReranker() if reranker is None else reranker

        if default_top_k < 1:
            raise ValueError("default_top_k must be greater than 0")

        if len(indexed_fields) == 0:
            raise ValueError("Indexed_fields must contain at least one field")

        if 'text' in indexed_fields:
            raise ValueError("The 'text' field cannot be used for metadata filtering. "
                             "Please remove it from indexed_fields")

    def query(self, queries: List[Query],
              global_metadata_filter: Optional[dict] = None
              ) -> List[QueryResult]:

        # Encode queries
        queries: List[KBQuery] = self._encoder.encode_queries(queries)

        # TODO: perform the actual index querying
        results: List[KBQueryResult]

        # Rerank results
        results = self._reranker.rerank(results)

        # Convert to QueryResult
        return [
            QueryResult(**r.dict(exclude={'values', 'sprase_values'})) for r in results
        ]
