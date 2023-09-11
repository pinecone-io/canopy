from typing import Union, Iterable, Optional, Any, Dict, List

import openai
import json
from context_engine.llm import BaseLLM
from context_engine.llm.models import Function, ModelParams
from context_engine.models.api_models import ChatResponse, StreamingChatChunk
from context_engine.models.data_models import Messages, Query


class OpenAILLM(BaseLLM):

    def __init__(self,
                 model_name: str,
                 *,
                 model_params: Optional[ModelParams] = None,
                 ):
        super().__init__(model_name,
                         model_params=model_params)
        self.available_models = [k["id"] for k in openai.Model.list().data]
        if model_name not in self.available_models:
            raise ValueError(
                f"Model {model_name} not found. " +
                " Available models: {self.available_models}"
            )

    def chat_completion(self,
                        messages: Messages,
                        *,
                        stream: bool = False,
                        max_tokens: Optional[int] = None,
                        model_params: Optional[ModelParams] = None,
                        ) -> Union[ChatResponse, Iterable[StreamingChatChunk]]:

        model_params_dict: Dict[str, Any] = {}
        model_params_dict.update(
            **self.default_model_params.dict(exclude_defaults=True)
        )
        if model_params:
            model_params_dict.update(**model_params.dict(exclude_defaults=True))

        messages = [m.dict() for m in messages]
        response = openai.ChatCompletion.create(model=self.model_name,
                                                messages=messages,
                                                stream=stream,
                                                max_tokens=max_tokens,
                                                **model_params_dict)

        def streaming_iterator(response):
            for chunk in response:
                yield StreamingChatChunk(**chunk)

        if stream:
            return streaming_iterator(response)

        return ChatResponse(**response)

    def enforced_function_call(self,
                               messages: Messages,
                               function: Function,
                               *,
                               max_tokens: Optional[int] = None,
                               model_params: Optional[ModelParams] = None) -> dict:
        # this enforces the model to call the function
        function_call = {"name": function.name}

        model_params_dict: Dict[str, Any] = {}
        model_params_dict.update(
            **self.default_model_params.dict(exclude_defaults=True)
        )
        if model_params:
            model_params_dict.update(**model_params.dict(exclude_defaults=True))

        messages = [m.dict() for m in messages]

        chat_completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            functions=[function.dict()],
            function_call=function_call,
            max_tokens=max_tokens,
            **model_params_dict
        )

        result = chat_completion.choices[0].message.function_call
        return json.loads(result["arguments"])

    async def achat_completion(self,
                               messages: Messages, *, stream: bool = False,
                               max_generated_tokens: Optional[int] = None,
                               model_params: Optional[ModelParams] = None
                               ) -> Union[ChatResponse,
                                          Iterable[StreamingChatChunk]]:
        raise NotImplementedError()

    async def agenerate_queries(self,
                                messages: Messages,
                                *,
                                max_generated_tokens: Optional[int] = None,
                                model_params: Optional[ModelParams] = None
                                ) -> List[Query]:
        raise NotImplementedError()
