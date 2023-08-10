from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict

from context_engine.knoweldge_base.models import QueryResult
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
    ) -> List[QueryResult]:
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