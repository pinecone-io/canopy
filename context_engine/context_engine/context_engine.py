from abc import ABC, abstractmethod
from typing import List

from context_engine.knoweldge_base.knoweldge_base import KnowledgeBase
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
             knowledge_base: KnowledgeBase,
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