import os
from abc import ABC, abstractmethod
from typing import List, Optional

from resin.context_engine.context_builder import StuffingContextBuilder
from resin.context_engine.context_builder.base import BaseContextBuilder
from resin.knoweldge_base.base import BaseKnowledgeBase
from resin.models.data_models import Context, Query

CE_DEBUG_INFO = os.getenv("CE_DEBUG_INFO", "FALSE").lower() == "true"


class BaseContextEngine(ABC):

    @abstractmethod
    def query(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        pass

    @abstractmethod
    async def aquery(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        pass


class ContextEngine(BaseContextEngine):

    DEFAULT_CONTEXT_BUILDER = StuffingContextBuilder

    def __init__(self,
                 knowledge_base: BaseKnowledgeBase,
                 *,
                 context_builder: Optional[BaseContextBuilder] = None,
                 global_metadata_filter: Optional[dict] = None
                 ):
        self.knowledge_base = knowledge_base
        self.context_builder = context_builder if context_builder is not None else \
            self.DEFAULT_CONTEXT_BUILDER()
        self.global_metadata_filter = global_metadata_filter

    def query(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        query_results = self.knowledge_base.query(
            queries,
            global_metadata_filter=self.global_metadata_filter)
        context = self.context_builder.build(query_results, max_context_tokens)

        if CE_DEBUG_INFO:
            context.debug_info["query_results"] = [qr.dict() for qr in query_results]
        return context

    async def aquery(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        raise NotImplementedError()
