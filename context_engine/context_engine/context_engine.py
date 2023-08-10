from abc import ABC, abstractmethod
from typing import List

from context_engine.context_engine.context_builders.base_context_builder import BaseContextBuilder
from context_engine.context_engine.context_builders.cb_types import CONTEXT_BUILDER_TYPES
from context_engine.knoweldge_base.knoweldge_base import KnowledgeBase
from context_engine.models.data_models import Context, Query
from context_engine.utils import type_from_str


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
             global_metadata_filter: dict = None,
             ):
    self.knowledge_base = knowledge_base
    self.global_metadata_filter = global_metadata_filter

    if context_builder_params is None:
        context_builder_params = {}
    context_builder_type = type_from_str(context_builder, CONTEXT_BUILDER_TYPES, "context builder")
    self.context_builder: BaseContextBuilder = context_builder_type(tokenizer=self.knowledge_base.tokenizer,
                                                                    **context_builder_params)


class ContextEngine(BaseContextEngine):

    def query(self,
              queries: List[Query],
              max_context_tokens: int,
    ) -> Context:
        query_results = self.knowledge_base.query(queries,
                                                  global_metadata_filter=self.global_metadata_filter)
        context = self.context_builder.build_context(query_results, max_context_tokens)
        return context

    async def aquery(self,
                     queries: List[Query],
                     max_context_tokens: int,
    ) -> Context:
        raise NotImplementedError