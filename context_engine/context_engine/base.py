from abc import ABC, abstractmethod
from typing import List, Optional

from context_engine.context_engine.context_builder.base import BaseContextBuilder
from context_engine.knoweldge_base.base_knoweldge_base import BaseKnowledgeBase
from context_engine.models.data_models import Context, Query


class BaseContextEngine(ABC):

    @abstractmethod
    def query(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        pass

    @abstractmethod
    async def aquery(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        pass


class ContextEngine(BaseContextEngine):

    def __init__(self,
                 knowledge_base: BaseKnowledgeBase,
                 context_builder: BaseContextBuilder,
                 *,
                 global_metadata_filter: Optional[dict] = None):
        self.knowledge_base = knowledge_base
        self.context_builder = context_builder
        self.global_metadata_filter = global_metadata_filter

    def query(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        query_results = self.knowledge_base.query(queries, global_metadata_filter=self.global_metadata_filter)
        context = self.context_builder.build(query_results, max_context_tokens)
        return context

    async def aquery(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        raise NotImplementedError
