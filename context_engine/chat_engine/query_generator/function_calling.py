from typing import List, Optional

from context_engine.chat_engine.prompt_builder.base import PromptBuilder
from context_engine.chat_engine.history_builder import RaisingHistoryBuilder
from context_engine.chat_engine.query_generator.base import QueryGenerator
from context_engine.knoweldge_base.tokenizer.base import Tokenizer
from context_engine.llm.base import BaseLLM
from context_engine.llm.models import (Function, FunctionParameters,
                                       FunctionArrayProperty)
from context_engine.models.data_models import Messages, Query

DEFAULT_SYSTEM_PROMPT = """Your task is to formulate search queries for a search engine,
to assist in responding to the user's question.
You should break down complex questions into sub-queries if needed."""

DEFAULT_FUNCTION_DESCRIPTION = """Query search engine for relevant information"""


class FunctionCallingQueryGenerator(QueryGenerator):

    def __init__(self,
                 *,
                 llm: BaseLLM,
                 top_k: int,
                 tokenizer: Tokenizer,  # TODO: need to remove this dependency
                 prompt: Optional[str] = None,
                 function_description: Optional[str] = None):
        self._llm = llm
        self._top_k = top_k
        self._system_prompt = prompt or DEFAULT_SYSTEM_PROMPT
        self._function_description = \
            function_description or DEFAULT_FUNCTION_DESCRIPTION

        # TODO: hardcoded for now, need to make it configurable
        history_prunner = RaisingHistoryBuilder(tokenizer)
        self._prompt_builder = PromptBuilder(tokenizer, history_prunner)

    def generate(self,
                 messages: Messages,
                 max_prompt_tokens: int) -> List[Query]:
        messages = self._prompt_builder.build(system_prompt=self._system_prompt,
                                              history=messages,
                                              max_tokens=max_prompt_tokens)
        arguments = self._llm.enforced_function_call(messages,
                                                     function=self._function)

        return [Query(text=q,
                      top_k=self._top_k,
                      metadata_filter=None)
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
