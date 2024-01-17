from abc import ABC, abstractmethod
from typing import Union, Iterable, Optional

from canopy.llm.models import Function
from canopy.models.api_models import ChatResponse, StreamingChatChunk
from canopy.models.data_models import Messages, Context
from canopy.utils.config import ConfigurableMixin


class BaseLLM(ABC, ConfigurableMixin):
    def __init__(self,
                 model_name: str):
        self.model_name = model_name

    @abstractmethod
    def chat_completion(self,
                        system_prompt: str,
                        chat_history: Messages,
                        context: Optional[Context] = None,
                        *,
                        stream: bool = False,
                        max_tokens: Optional[int] = None,
                        model_params: Optional[dict] = None,
                        ) -> Union[ChatResponse, Iterable[StreamingChatChunk]]:
        pass

    @abstractmethod
    def enforced_function_call(self,
                               system_prompt: str,
                               chat_history: Messages,
                               function: Function,
                               *,
                               max_tokens: Optional[int] = None,
                               model_params: Optional[dict] = None,
                               ) -> dict:
        pass

    @abstractmethod
    async def achat_completion(self,
                               system_prompt: str,
                               chat_history: Messages,
                               context: Optional[Context] = None,
                               *,
                               stream: bool = False,
                               max_generated_tokens: Optional[int] = None,
                               model_params: Optional[dict] = None,
                               ) -> Union[ChatResponse,
                                          Iterable[StreamingChatChunk]]:
        pass

    @abstractmethod
    async def aenforced_function_call(self,
                                      system_prompt: str,
                                      chat_history: Messages,
                                      function: Function,
                                      *,
                                      max_tokens: Optional[int] = None,
                                      model_params: Optional[dict] = None
                                      ):
        pass
