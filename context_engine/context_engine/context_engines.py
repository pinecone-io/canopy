from abc import ABC, abstractmethod
from typing import List

from context_engine.models.data_models import Context, Query


class BaseContextEngine(ABC):


    @abstractmethod
    def query(self,
              queries: List[Query],
              max_context_tokens: int,
    ) -> Context:
        pass

    @abstractmethod
    async def aquery(self,
                     queries: List[Query],
                     max_context_tokens: int,
    ) -> Context:
        pass


def __init__(self,
             *,
             index_name: str,
             embedding: str = "OpenAI/ada-002",
             tokenization: str = "OpenAI/gpt-3.5-turbo-0613",
             knowledge_base: str = "Pinecone",
             knowledge_base_params: dict = None,
             context_builder: str = "stuffing",
             context_builder_params: dict = None,
             ):
    pass
class ContextEngine(BaseContextEngine):

    def query(self,
              queries: List[Query],
              max_context_tokens: int,
    ) -> Context:
        raise NotImplementedError

    async def aquery(self,
                     queries: List[Query],
                     max_context_tokens: int,
    ) -> Context:
        raise NotImplementedError