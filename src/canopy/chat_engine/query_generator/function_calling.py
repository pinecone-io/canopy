from typing import List, Optional

from canopy.chat_engine.history_pruner import RaisingHistoryPruner
from canopy.chat_engine.query_generator import QueryGenerator
from canopy.llm import BaseLLM, OpenAILLM
from canopy.llm.models import (Function, FunctionParameters,
                               FunctionArrayProperty)
from canopy.models.data_models import Messages, Query

DEFAULT_SYSTEM_PROMPT = """Your task is to formulate search queries for a search engine, to assist in responding to the user's question.
You should break down complex questions into sub-queries if needed.
"""  # noqa: E501

DEFAULT_FUNCTION_DESCRIPTION = """Query search engine for relevant information"""


class FunctionCallingQueryGenerator(QueryGenerator):

    _DEFAULT_COMPONENTS = {
        "llm": OpenAILLM,
    }

    def __init__(self,
                 *,
                 llm: Optional[BaseLLM] = None,
                 prompt: Optional[str] = None,
                 function_description: Optional[str] = None):
        self._llm = llm or self._DEFAULT_COMPONENTS["llm"]()
        self._system_prompt = prompt or DEFAULT_SYSTEM_PROMPT
        self._function_description = \
            function_description or DEFAULT_FUNCTION_DESCRIPTION
        self._history_pruner = RaisingHistoryPruner()

    def generate(self,
                 messages: Messages,
                 max_prompt_tokens: int) -> List[Query]:
        messages = self._history_pruner.build(system_prompt=self._system_prompt,
                                              chat_history=messages,
                                              max_tokens=max_prompt_tokens)
        try:
            arguments = self._llm.enforced_function_call(
                system_prompt=self._system_prompt,
                chat_history=messages,
                function=self._function
            )
        except NotImplementedError as e:
            raise RuntimeError(
                "FunctionCallingQueryGenerator requires an LLM that supports "
                "function calling. Please provide a different LLM, "
                "or alternatively select a different QueryGenerator class. "
                f"Received the following error from LLM:\n{e}"
            ) from e

        return [Query(text=q)
                for q in arguments["queries"]]

    async def agenerate(self,
                        messages: Messages,
                        max_prompt_tokens: int) -> List[Query]:
        raise NotImplementedError

    @property
    def _function(self) -> Function:
        return Function(
            name="query_knowledgebase",
            description=self._function_description,
            parameters=FunctionParameters(
                required_properties=[
                    FunctionArrayProperty(
                        name="queries",
                        items_type="string",
                        description='List of queries to send to the search engine.',
                    ),
                ]
            ),
        )
