import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from context_engine.context_engine.context_builder import CONTEXT_BUILDER_CLASSES
from context_engine.context_engine.context_builder.base import ContextBuilder
from context_engine.knoweldge_base.base import BaseKnowledgeBase
from context_engine.models.data_models import Context, Query
from context_engine.utils import initialize_from_config

logger = logging.getLogger(__name__)


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
                 *,
                 context_builder: Optional[ContextBuilder] = None,
                 global_metadata_filter: Optional[dict] = None
                 ):
        self.knowledge_base = knowledge_base
        if context_builder is None:
            default_type = CONTEXT_BUILDER_CLASSES['default']
            logger.info(f"Initializing ContextEngine with default context builder "
                        f"{default_type.__name__}")
            self.context_builder = default_type()
        else:
            if not isinstance(context_builder, ContextBuilder):
                raise ValueError(f"context_builder must be an instance of "
                                 f"ContextBuilder, got {type(context_builder)}")
            self.context_builder = context_builder

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
    def from_config(cls,
                    config: dict,
                    *,
                    knowledge_base: BaseKnowledgeBase,
                    context_builder: Optional[ContextBuilder] = None):

        context_builder_config = config.pop("context_builder", None)
        if context_builder and context_builder_config:
            raise ValueError("Cannot provide both context_builder override and "
                             "context_builder config. If you wish to override with your"
                             " own context_builder, remove the 'context_builder' "
                             "key from the config")
        if context_builder_config:
            context_builder = initialize_from_config(context_builder_config,
                                                     CONTEXT_BUILDER_CLASSES,
                                                     "context_builder")
        return cls(knowledge_base,
                   context_builder=context_builder,
                   **config)
