from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict

from context_engine.knoweldge_base.models import KBQueryResult
from context_engine.knoweldge_base.tokenizers.base_tokenizer import Tokenizer
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
    ) -> List[QueryResult]:
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
                 tokenization: str = "OpenAI/gpt-3.5-turbo-0613",
                 sparse_encoding: str = "None",
                 chunking: str = "markdown",
                 chunk_size: int = 200,
                 ranking: str = "None",
                 **kwargs
                 ):
        pass



