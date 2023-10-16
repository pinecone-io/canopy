from abc import ABC, abstractmethod
from typing import List, Optional

from resin.knoweldge_base.models import QueryResult
from resin.models.data_models import Query, Document
from resin.utils.config import ConfigurableMixin


class BaseKnowledgeBase(ABC, ConfigurableMixin):
    """
    KnowledgeBase is an abstract class that defines the interface for a knowledge base.
    """

    @abstractmethod
    def query(self,
              queries: List[Query],
              global_metadata_filter: Optional[dict] = None
              ) -> List[QueryResult]:
        """
        Query the knowledge base.

        Args:
            queries: A list of queries to run against the knowledge base.
            global_metadata_filter: A metadata filter to apply to all queries.
                                    in addition to any query-specific filters.
        Returns:
            A list of QueryResult objects.
        """
        pass

    @abstractmethod
    def upsert(self,
               documents: List[Document],
               namespace: str = "", ) -> None:
        """
        Upsert documents into the knowledge base.

        Args:
            documents: A list of documents to upsert.
            namespace: The namespace to upsert the documents into.

        Returns:
            None
        """
        pass

    # TODO: Do we want delete by metadata?
    @abstractmethod
    def delete(self,
               document_ids: List[str],
               namespace: str = "") -> None:
        """
        Delete documents from the knowledge base.

        Args:
            document_ids: A list of document ids to delete.
            namespace: The namespace to delete the documents from.

        Returns:
            None
        """
        pass

    @abstractmethod
    def verify_index_connection(self) -> None:
        """
        Verify that the knowledge base is connected
        and the index is ready to be queried.

        Returns:
            None if the index is connected correctly, otherwise raises an exception.
        """
        pass

    @abstractmethod
    async def aquery(self,
                     queries: List[Query],
                     global_metadata_filter: Optional[dict] = None
                     ) -> List[QueryResult]:
        """
        Async version of query the knowledge base.

        Args:
            queries: A list of queries to run against the knowledge base.
            global_metadata_filter: A metadata filter to apply to all queries.
                                    in addition to any query-specific filters.

        Returns:
            A list of QueryResult objects.
        """
        pass

    @abstractmethod
    async def aupsert(self,
                      documents: List[Document],
                      namespace: str = "",
                      ) -> None:
        """
        Async version of upsert documents into the knowledge base.

        Args:
            documents: A list of documents to upsert.
            namespace: The namespace to upsert the documents into.

        Returns:
            None
        """
        pass

    @abstractmethod
    async def adelete(self,
                      document_ids: List[str],
                      namespace: str = "") -> None:
        """
        Async version of delete documents from the knowledge base.

        Args:
            document_ids: A list of document ids to delete.
            namespace: The namespace to delete the documents from.

        Returns:
            None
        """
        pass
