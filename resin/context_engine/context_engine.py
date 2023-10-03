import os
from abc import ABC, abstractmethod
from typing import List, Optional

from resin.context_engine.context_builder import StuffingContextBuilder
from resin.context_engine.context_builder.base import ContextBuilder
from resin.knoweldge_base import KnowledgeBase
from resin.knoweldge_base.base import BaseKnowledgeBase
from resin.models.data_models import Context, Query
from resin.utils.config import ConfigurableMixin, FactoryMixin

CE_DEBUG_INFO = os.getenv("CE_DEBUG_INFO", "FALSE").lower() == "true"


class BaseContextEngine(ABC, FactoryMixin):

    @abstractmethod
    def query(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        pass

    @abstractmethod
    async def aquery(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        pass


class ContextEngine(BaseContextEngine, ConfigurableMixin):

    _DEFAULT_COMPONENTS = {
        'knowledge_base': KnowledgeBase,
        'context_builder': StuffingContextBuilder,
    }


    def __init__(self,
                 *,
                 knowledge_base: Optional[BaseKnowledgeBase] = None,
                 context_builder: Optional[ContextBuilder] = None,
                 global_metadata_filter: Optional[dict] = None
                 ):
        self.knowledge_base = self._set_component(
            BaseKnowledgeBase, 'knowledge_base', knowledge_base)
        self.context_builder = self._set_component(
            ContextBuilder, 'context_builder', context_builder)
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

    @classmethod
    def from_config(cls, config: dict):
        return cls._from_config(config=config)