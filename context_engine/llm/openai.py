from typing import Union, Iterable, Optional, Any, Dict

import openai

from context_engine.llm.base import LLM
from context_engine.llm.models import Function, ModelParams
from context_engine.models.api_models import ChatResponse, StreamingChatResponse
from context_engine.models.data_models import History, LLMResponse


class OpenAILLM(LLM):

    def chat_completion(self,
                        messages: History,
                        *,
                        stream: bool = False,
                        max_generated_tokens: Optional[int] = None,
                        model_params: Optional[ModelParams] = None,
                        ) -> Union[ChatResponse, Iterable[StreamingChatResponse]]:

        model_params_dict: Dict[str, Any] = {}
        if model_params:
            model_params_dict.update(**model_params.dict(exclude_defaults=True))
        model_params_dict.update(**self.default_model_params.dict())
        model_params_dict['n'] = model_params_dict.pop('num_choices')

        max_generated_tokens = max_generated_tokens or self.default_max_generated_tokens

        response = openai.ChatCompletion.create(model=self.model_name,
                                                messages=messages,
                                                stream=stream,
                                                max_tokens=max_generated_tokens,
                                                **model_params_dict)

        def streaming_iterator(response):
            for chunk in response:
                yield StreamingChatResponse(**chunk)

        if stream:
            return streaming_iterator(response)

        return ChatResponse(**response)
