from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict

from context_engine.models.data_models import Query, QueryResult, Document


class PineconeKnowledgeBase:
    def __init__(self,
                 *,
                 index_name: str,
                 embedding: str,
                 tokenization: str,
                 sparse_embedding: str = "None",
                 ranking: str = "None",
                 **kwargs
                 ):
        pass

    def query(self,
              queries: List[Query],
              global_metadata_filter: Optional[dict] = None
    ) -> List[QueryResult]:
        pass

    def upsert(self,
               documents: List[Union[Dict[str, Union[str, dict]], Document]],
               namespace: str = "",

    ) -> None:
        pass
