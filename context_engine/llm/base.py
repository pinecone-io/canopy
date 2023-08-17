from abc import ABC, abstractmethod
from typing import Union, Iterable

from context_engine.llm.models import AssistantMessage, Function
from context_engine.models.data_models import History, LLMResponse


class LLM(ABC):
    def __init__(self,
                 model_name: str,
                 max_generated_tokens: int,
                 ):
        self.model_name = model_name
        self.max_generated_tokens = max_generated_tokens

    @abstractmethod
    def chat_completion(self,
                        messages: History,
                        stream: bool = False,
                        ) -> Union[LLMResponse, Iterable[LLMResponse]]:
        pass

    @abstractmethod
    def enforced_function_call(self,
                               messages: History,
                               function: Function,
                               ) -> dict:
        pass