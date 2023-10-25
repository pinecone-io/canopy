from abc import ABC, abstractmethod
from typing import Union, Iterable, Optional, List

from canopy.llm.models import Function, ModelParams
from canopy.models.api_models import ChatResponse, StreamingChatChunk
from canopy.models.data_models import Messages, Query
from canopy.utils.config import ConfigurableMixin


class BaseLLM(ABC, ConfigurableMixin):
    def __init__(self,
                 model_name: str,
                 *,
                 model_params: Optional[ModelParams] = None,
                 ):
        self.model_name = model_name
        # TODO: consider removing altogether
        self.default_model_params = model_params or ModelParams()

    @abstractmethod
    def chat_completion(self,
                        messages: Messages,
                        *,
                        stream: bool = False,
                        max_tokens: Optional[int] = None,
                        model_params: Optional[ModelParams] = None,
                        ) -> Union[ChatResponse, Iterable[StreamingChatChunk]]:
        pass

    @abstractmethod
    def enforced_function_call(self,
                               messages: Messages,
                               function: Function,
                               *,
                               max_tokens: Optional[int] = None,
                               model_params: Optional[ModelParams] = None
                               ) -> dict:
        pass

    @abstractmethod
    async def achat_completion(self,
                               messages: Messages,
                               *,
                               stream: bool = False,
                               max_generated_tokens: Optional[int] = None,
                               model_params: Optional[ModelParams] = None,
                               ) -> Union[ChatResponse,
                                          Iterable[StreamingChatChunk]]:
        pass

    @abstractmethod
    async def agenerate_queries(self,
                                messages: Messages,
                                *,
                                max_generated_tokens: Optional[int] = None,
                                model_params: Optional[ModelParams] = None,
                                ) -> List[Query]:
        pass
