from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict

from context_engine.knoweldge_base.encoders.base_encoder import BaseEncoder
from context_engine.knoweldge_base.models import KBQueryResult
from context_engine.knoweldge_base.tokenizers.base_tokenizer import Tokenizer
from context_engine.knoweldge_base.kb_types import TOKENIZER_TYPES, type_from_str, CHUNKER_TYPES, RERANKER_TYPES
from context_engine.models.data_models import Query, Document


class KnowledgeBase(ABC):
    """
    KnowledgeBase is an abstract class that defines the interface for a knowledge base.
    """
    @abstractmethod
    def query(self,
              queries: List[Query],
              global_metadata_filter: Optional[dict] = None,
    ) -> List[KBQueryResult]:
        pass

    @abstractmethod
    def upsert(self,
               documents: List[Union[Dict[str, Union[str, dict]], Document]],
               namespace: str = "",

    ) -> None:
        pass

    # TODO: Do we want delete by metadata?
    @abstractmethod
    def delete(self,
               document_ids: List[str],
               namespace: str = "",
    ) -> None:
        pass

    @abstractmethod
    async def aquery(self,
                     queries: List[Query],
                     global_metadata_filter: Optional[dict] = None,
    ) -> List[KBQueryResult]:
        pass


    @abstractmethod
    async def aupsert(self,
                      documents: List[Union[Dict[str, Union[str, dict]], Document]],
                      namespace: str = "",

    ) -> None:
        pass

    @abstractmethod
    async def adelete(self,
               document_ids: List[str],
               namespace: str = "",
    ) -> None:
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> Tokenizer:
        pass


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
