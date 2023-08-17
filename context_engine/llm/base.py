from abc import ABC, abstractmethod
from typing import Union, Iterable, Optional

from context_engine.llm.models import Function, ModelParams
from context_engine.models.data_models import History, LLMResponse


class LLM(ABC):
    def __init__(self,
                 model_name: str,
                 max_generated_tokens: int,
                 *,
                 model_params: Optional[ModelParams] = None,
                 ):
        self.model_name = model_name
        self.default_max_generated_tokens = max_generated_tokens
        self.default_model_params = model_params or ModelParams()

    @abstractmethod
    def chat_completion(self,
                        messages: History,
                        *,
                        stream: bool = False,
                        max_generated_tokens: Optional[int] = None,
                        model_params: Optional[ModelParams] = None,
                        ) -> Union[LLMResponse, Iterable[LLMResponse]]:
        pass

    @abstractmethod
    def enforced_function_call(self,
                               messages: History,
                               function: Function,
                               max_generated_tokens: Optional[int] = None,
                               model_params: Optional[ModelParams] = None,
                               ) -> dict:
        pass

    @abstractmethod
    async def achat_completion(self,
                               messages: History,
                               *,
                               stream: bool = False,
                               max_generated_tokens: Optional[int] = None,
                               model_params: Optional[ModelParams] = None,
                               ) -> Union[LLMResponse, Iterable[LLMResponse]]:
        pass

    @abstractmethod
    async def aenforced_function_call(self,
                                      messages: History,
                                      function: Function,
                                      *,
                                      max_generated_tokens: Optional[int] = None,
                                      model_params: Optional[ModelParams] = None,
                                      ) -> dict:
        pass
