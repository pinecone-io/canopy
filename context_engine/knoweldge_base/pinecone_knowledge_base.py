from typing import List, Optional

from context_engine.knoweldge_base.encoders.base_encoder import BaseEncoder
from context_engine.knoweldge_base.kb_types import type_from_str, TOKENIZER_TYPES, CHUNKER_TYPES, RERANKER_TYPES
from context_engine.knoweldge_base.tokenizers.base_tokenizer import Tokenizer

from context_engine.models.data_models import Query, Document
from context_engine.knoweldge_base.models import KBQueryResult

class PineconeKnowledgeBase:
    def __init__(self,
                 *,
                 index_name: str,
                 embedding: str = "OpenAI/ada-002",
                 sparse_encoding: str = "None",
                 tokenization: str = "OpenAI/gpt-3.5-turbo-0613",
                 chunking: str = "markdown",
                 reranking: str = "None",
                 **kwargs
                 ):

        self.index_name = index_name

        # TODO: decide how we are instantiating the encoder - as a single encoder that does both dense and spars
        # or as two separate encoders
        self._encoder: BaseEncoder

        # Instantiate tokenizer
        try:
            tokenizer_type, tokenizer_model_name = tokenization.split("/")
        except ValueError as e:
            raise ValueError("tokenization must be in the format <tokenizer_type>/<tokenizer_model_name>") from e

        tokenizer_type = type_from_str(tokenizer_type, TOKENIZER_TYPES, "tokenization")
        self._tokenizer: Tokenizer = tokenizer_type(tokenizer_model_name, **kwargs)

        # Instantiate chunker
        self._chunker = type_from_str(chunking, CHUNKER_TYPES, "chunking")(**kwargs)

        # Instantiate reranker
        self._reranker = type_from_str(reranking, RERANKER_TYPES, "Reranking")(**kwargs)


# TODO: remove, for testing only
if __name__ == "__main__":
    pc = PineconeKnowledgeBase(index_name="test")
    print(pc)
