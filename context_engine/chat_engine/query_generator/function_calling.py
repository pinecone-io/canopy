from typing import List, Optional

from context_engine.chat_engine.history_builder.base import BaseHistoryBuilder
from context_engine.chat_engine.query_generator.base import QueryGenerator
from context_engine.llm.base import BaseLLM
from context_engine.llm.models import ModelParams, Function, FunctionParameters, FunctionArrayProperty
from context_engine.models.data_models import Messages, Query, Role, MessageBase

DEFAULT_SYSTEM_PROMPT = """Your task is to formulate search queries for a search engine,
to assist in responding to the user's question. You should break down complex questions into sub-queries if needed."""
DEFAULT_FUNCTION_DESCRIPTION = """Query search engine for relevant information"""


class FunctionCallingQueryGenerator(QueryGenerator):

    def __init__(self,
                 llm: BaseLLM,
                 history_builder: BaseHistoryBuilder,
                 top_k: float,
                 system_prompt: Optional[str] = None,
                 function_description: Optional[str] = None,
                 model_params: Optional[ModelParams] = None,):
        self._llm = llm
        self._history_builder = history_builder
        self._top_k = top_k
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._function_description = function_description or DEFAULT_FUNCTION_DESCRIPTION
        self._model_params = model_params

    def generate(self,
                 messages: Messages,
                 max_prompt_tokens: int) -> List[Query]:
        system_message = MessageBase(role=Role.SYSTEM,
                                     content=self._system_prompt)
        max_history_prompt = max_prompt_tokens - len(system_message.json())
        history = self._history_builder.build(messages,
                                              max_tokens=max_prompt_tokens)
        messages = [system_message] + messages
        arguments = self._llm.enforced_function_call(messages,
                                                     function=self._function,
                                                     max_tokens=max_prompt_tokens,
                                                     model_params=self._model_params)

        return [Query(text=q, top_k=self._top_k) for q in arguments["queries"]]

    async def agenerate(self,
                        messages: Messages,
                        max_prompt_tokens: int) -> List[Query]:
        pass

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
                        description=f'List of queries to send to the search engine.',
                    ),
                ]
            ),
        )
