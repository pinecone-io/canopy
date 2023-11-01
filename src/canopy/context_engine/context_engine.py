import os
from abc import ABC, abstractmethod
from typing import List, Optional

from canopy.context_engine.context_builder import StuffingContextBuilder
from canopy.context_engine.context_builder.base import ContextBuilder
from canopy.knowledge_base import KnowledgeBase
from canopy.knowledge_base.base import BaseKnowledgeBase
from canopy.models.data_models import Context, Query
from canopy.utils.config import ConfigurableMixin

CE_DEBUG_INFO = os.getenv("CE_DEBUG_INFO", "FALSE").lower() == "true"


class BaseContextEngine(ABC, ConfigurableMixin):

    @abstractmethod
    def query(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        pass

    @abstractmethod
    async def aquery(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        pass


class ContextEngine(BaseContextEngine):
    """
    ContextEngine is responsible for providing context to the LLM, given a set of search queries.

    Once called with a set of queries, the ContextEngine will go through the following steps:
    1. Query the knowledge base for relevant documents
    2. Build a context from the documents retrieved that can be injected into the LLM prompt

    The context engine considers token budgeting when building the context, and tries to maximize the amount of relevant information that can be provided to the LLM within the token budget.

    To create a context engine, you must provide a knowledge base and optionally a context builder.

    Example:
    >>> from canopy.context_engine import ContextEngine
    >>> from canopy.models.data_models import Query
    >>> context_engine = ContextEngine(knowledge_base=knowledge_base)
    >>> context_engine.query(Query(text="What is the capital of France?"), max_context_tokens=1000)

    To create a knowledge base, see the documentation for the knowledge base module (canopy.knowledge_base.knowledge_base).
    """  # noqa: E501

    _DEFAULT_COMPONENTS = {
        'knowledge_base': KnowledgeBase,
        'context_builder': StuffingContextBuilder,
    }

    def __init__(self,
                 knowledge_base: BaseKnowledgeBase,
                 *,
                 context_builder: Optional[ContextBuilder] = None,
                 global_metadata_filter: Optional[dict] = None
                 ):
        """
        Initialize a new ContextEngine.

        Args:
            knowledge_base: The knowledge base to query for retrieving documents
            context_builder: The context builder to use for building the context from the retrieved documents. Defaults to `StuffingContextBuilder`
            global_metadata_filter: A metadata filter to apply to all queries. See: https://docs.pinecone.io/docs/metadata-filtering
        """  # noqa: E501

        if not isinstance(knowledge_base, BaseKnowledgeBase):
            raise TypeError("knowledge_base must be an instance of BaseKnowledgeBase, "
                            f"not {type(self.knowledge_base)}")
        self.knowledge_base = knowledge_base

        if context_builder:
            if not isinstance(context_builder, ContextBuilder):
                raise TypeError(
                    "context_builder must be an instance of ContextBuilder, "
                    f"not {type(context_builder)}"
                )
            self.context_builder = context_builder
        else:
            self.context_builder = self._DEFAULT_COMPONENTS['context_builder']()

        self.global_metadata_filter = global_metadata_filter

    def query(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        """
        Query the knowledge base for relevant documents and build a context from the retrieved documents that can be injected into the LLM prompt.

        Args:
            queries: A list of queries to use for retrieving documents from the knowledge base
            max_context_tokens: The maximum number of tokens to use for the context

        Returns:
            A Context object containing the retrieved documents and metadata

        Example:
        >>> from canopy.context_engine import ContextEngine
        >>> from canopy.models.data_models import Query
        >>> context_engine = ContextEngine(knowledge_base=knowledge_base)
        >>> context_engine.query(Query(text="What is the capital of France?"), max_context_tokens=1000)
        """  # noqa: E501
        query_results = self.knowledge_base.query(
            queries,
            global_metadata_filter=self.global_metadata_filter)
        context = self.context_builder.build(query_results, max_context_tokens)

        if CE_DEBUG_INFO:
            context.debug_info["query_results"] = [qr.dict() for qr in query_results]
        return context

    async def aquery(self, queries: List[Query], max_context_tokens: int, ) -> Context:
        raise NotImplementedError()
