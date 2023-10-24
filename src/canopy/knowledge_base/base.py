from abc import ABC, abstractmethod
from typing import List, Optional

from canopy.knowledge_base.models import QueryResult
from canopy.models.data_models import Query, Document
from canopy.utils.config import ConfigurableMixin


class BaseKnowledgeBase(ABC, ConfigurableMixin):
    """
    KnowledgeBase is an abstract class that defines the interface for a knowledge base.
    """

    @abstractmethod
    def query(self,
              queries: List[Query],
              global_metadata_filter: Optional[dict] = None
              ) -> List[QueryResult]:
        pass

    @abstractmethod
    def upsert(self,
               documents: List[Document],
               namespace: str = "", ) -> None:
        pass

    # TODO: Do we want delete by metadata?
    @abstractmethod
    def delete(self,
               document_ids: List[str],
               namespace: str = "") -> None:
        pass

    @abstractmethod
    def verify_index_connection(self) -> None:
        pass

    @abstractmethod
    async def aquery(self,
                     queries: List[Query],
                     global_metadata_filter: Optional[dict] = None
                     ) -> List[QueryResult]:
        pass

    @abstractmethod
    async def aupsert(self,
                      documents: List[Document],
                      namespace: str = "",
                      ) -> None:
        pass

    @abstractmethod
    async def adelete(self,
                      document_ids: List[str],
                      namespace: str = "") -> None:
        pass
