from abc import ABC, abstractmethod
from typing import List

from context_engine.chat_engine.models import HistoryPruningMethod
from context_engine.chat_engine.prompt_builder import PromptBuilder
from context_engine.llm import BaseLLM
from context_engine.models.data_models import Messages, Query


class QueryGenerator(ABC):
    def __init__(self, *, llm: BaseLLM):
        self._llm = llm
        self._prompt_builder = PromptBuilder(HistoryPruningMethod.RAISE, 1)

    @abstractmethod
    def generate(self,
                 messages: Messages,
                 max_prompt_tokens: int,
                 ) -> List[Query]:
        pass

    @abstractmethod
    async def agenerate(self,
                        messages: Messages,
                        max_prompt_tokens: int,
                        ) -> List[Query]:
        raise NotImplementedError
