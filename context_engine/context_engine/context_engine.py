from abc import ABC, abstractmethod
from typing import List, Optional

from context_engine.context_engine.context_builder import StuffingContextBuilder
from context_engine.context_engine.context_builder.base import ContextBuilder
from context_engine.knoweldge_base.base import BaseKnowledgeBase
from context_engine.models.data_models import Context, Query
from context_engine.utils import get_class_from_config


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
                 context_builder: Optional[ContextBuilder] = None,
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
        return context

    async def aquery(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: dict,
                    knowledge_base: BaseKnowledgeBase,
                    context_builder: Optional[ContextBuilder] = None):
        if context_builder is None:
            context_builder_cfg = config.get("context_builder", {})
            context_builder = get_class_from_config(config,
                                                    CONTEXT_BUILDER_CLASSES,
                                                    default=cls.DEFAULT_CONTEXT_BUILDER,
                                                    name="Context Builder")
