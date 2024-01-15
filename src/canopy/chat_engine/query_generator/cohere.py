from typing import List, Optional

from canopy.chat_engine.query_generator import QueryGenerator
from canopy.chat_engine.history_pruner.raising import RaisingHistoryPruner
from canopy.llm import BaseLLM, CohereLLM
from canopy.models.data_models import Messages, Query


class CohereQueryGenerator(QueryGenerator):
    """
    Query generator for LLM clients that have a built-in feature to
    generate search queries from chat messages.
    """
    _DEFAULT_COMPONENTS = {
        "llm": CohereLLM,
    }

    def __init__(self,
                 *,
                 llm: Optional[BaseLLM] = None):
        self._llm = llm or self._DEFAULT_COMPONENTS["llm"]()

        if not isinstance(self._llm, CohereLLM):
            raise NotImplementedError(
                "CohereQueryGenerator only compatible with CohereLLM"
            )

        self._history_pruner = RaisingHistoryPruner()

    def generate(self,
                 messages: Messages,
                 max_prompt_tokens: int) -> List[Query]:
        messages = self._history_pruner.build(chat_history=messages,
                                              max_tokens=max_prompt_tokens)
        queries = self._llm.generate_search_queries(  # type: ignore[union-attr]
            messages)
        return [Query(text=query) for query in queries]

    async def agenerate(self,
                        messages: Messages,
                        max_prompt_tokens: int) -> List[Query]:
        raise NotImplementedError
