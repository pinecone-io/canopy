from abc import ABC, abstractmethod
from typing import Union, Iterable, Optional, cast

from context_engine.llm.models import Function, ModelParams, UserMessage
from context_engine.models.api_models import ChatResponse, StreamingChatResponse
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
                        ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:
        pass

    def call(self,
             prompt: str,
             *,
             history: Optional[History],
             max_generated_tokens: Optional[int] = None,
             model_params: Optional[ModelParams] = None,
             ) -> LLMResponse:
        if not history:
            history = []
        messages: History = history + [UserMessage(content=prompt)]
        response = self.chat_completion(
            messages,
            stream=False,
            max_generated_tokens=max_generated_tokens,
            model_params=model_params
        )
        response = cast(ChatResponse, response)

        return LLMResponse(id=response.id,
                           choices=[
                               c.message.content for c in response.choices
                           ],
                           generated_tokens=response.usage.completion_tokens,
                           prompt_tokens=response.usage.prompt_tokens)

    @abstractmethod
    def enforced_function_call(self,
                               prompt: str,
                               function: Function,
                               *,
                               history: Optional[History],
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
