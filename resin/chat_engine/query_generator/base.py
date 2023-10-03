from abc import ABC, abstractmethod
from typing import List

from resin.chat_engine.models import HistoryPruningMethod
from resin.chat_engine.prompt_builder import PromptBuilder
from resin.llm import BaseLLM
from resin.models.data_models import Messages, Query
from resin.utils.config import FactoryMixin


class QueryGenerator(ABC, FactoryMixin):
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
