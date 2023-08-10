from typing import List, Optional

from context_engine.knoweldge_base.chunkers.base_chunker import Chunker
from context_engine.knoweldge_base.encoders.base_encoder import BaseEncoder
from context_engine.knoweldge_base.kb_types import TOKENIZER_TYPES, CHUNKER_TYPES, RERANKER_TYPES
from context_engine.knoweldge_base.models import KBQueryResult, KBQuery, QueryResult
from context_engine.knoweldge_base.rerankers.reranker import Reranker
from context_engine.knoweldge_base.tokenizers.base_tokenizer import Tokenizer
from context_engine.models.data_models import Query
from context_engine.utils import type_from_str


class PineconeKnowledgeBase:
    def __init__(self,
                 *,
                 index_name: str,
                 embedding: str = "OpenAI/ada-002",
                 sparse_encoding: str = "None",
                 tokenization: str = "OpenAI/gpt-3.5-turbo-0613",
                 chunking: str = "markdown",
                 reranking: str = "no_reranking",
                 **kwargs
                 ):

        self.index_name = index_name

        # TODO: decide how we are instantiating the encoder - as a single encoder that does both dense and
        #  spars
        # or as two separate encoders
        self._encoder: BaseEncoder

        # Instantiate tokenizer
        try:
            tokenizer_type_name, tokenizer_model_name = tokenization.split("/")
        except ValueError as e:
            raise ValueError("tokenization must be in the format <tokenizer_type>/<tokenizer_model_name>") \
                from e

        tokenizer_type = type_from_str(tokenizer_type_name, TOKENIZER_TYPES, "tokenization")
        self._tokenizer: Tokenizer = tokenizer_type(tokenizer_model_name, **kwargs)

        # Instantiate chunker
        self._chunker: Chunker = type_from_str(chunking, CHUNKER_TYPES, "chunking")(**kwargs)

        # Instantiate reranker
        self._reranker: Reranker = type_from_str(reranking, RERANKER_TYPES, "Reranking")(**kwargs)

    def query(self, queries: List[Query], global_metadata_filter: Optional[dict] = None, ) -> List[
        QueryResult]:

        # Convert to KBQuery, which also includes dense and sparse vectors
        queries: List[KBQuery] = [KBQuery(**q.dict()) for q in queries]

        # Encode queries
        queries = self._encoder.encode_queries(queries)

        # TODO: perform the actual index querying
        results: List[KBQueryResult]

        # Rerank results
        results = self._reranker.rerank_results(results)

        # Convert to QueryResult
        results: List[QueryResult] = [QueryResult(**r.dict(exclude={'values', 'sprase_values'})) for r in
                                      results]

        return results


# TODO: remove, for testing only
if __name__ == "__main__":
    query = Query(text="test query", namespace="test_namespace", metadata_filter={"test": "test"}, top_k=10)
    kbquery = KBQuery(**query.dict())
    print(id(kbquery.text) == id(query.text))

    pc = PineconeKnowledgeBase(index_name="test")
    print(pc)
