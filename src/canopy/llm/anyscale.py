from typing import Union, Iterable, Optional, Any, Dict, List

import jsonschema
import openai
import json
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from canopy.utils.openai_exceptions import OPEN_AI_TRANSIENT_EXCEPTIONS, is_openai_v1
from canopy.llm import BaseLLM
from canopy.llm.models import Function, ModelParams
from canopy.models.api_models import ChatResponse, StreamingChatChunk
from canopy.models.data_models import Messages, Query

ANYSCALE_API_BASE = "https://console.endpoints.anyscale.com/m/v1"
class AnyscaleLLM(BaseLLM):
    """
    Anyscale LLM wrapper built on top of the OpenAI Python client.

    Note: Anyscale requires a valid API key to use this class.
          You can set the "OPENAI_API_KEY" environment variable to your API key.
          Or you can directly set it as follows:
          >>> import openai
          >>> openai.api_key = "YOUR_API_KEY"
    """
    def __init__(self,
                 api_key,
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 *,
                 model_params: Optional[ModelParams] = None,
                 ):
        self.api_key = api_key
        super().__init__(model_name,
                         model_params=model_params)

    @property
    def available_models(self):
        if is_openai_v1():
            client = openai.OpenAI(
                base_url = ANYSCALE_API_BASE,
                api_key = self.api_key
            )
            return [k.id for k in client.models.list().data]
        else:
            return [k["id"] for k in openai.Model.list().data]

    @retry(
        reraise=True,
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(OPEN_AI_TRANSIENT_EXCEPTIONS),
    )
    def chat_completion(self,
                        messages: Messages,
                        *,
                        stream: bool = False,
                        max_tokens: Optional[int] = None,
                        model_params: Optional[ModelParams] = None,
                        ) -> Union[ChatResponse, Iterable[StreamingChatChunk]]:
        """
        Chat completion using the OpenAI API.

        Note: this function is wrapped in a retry decorator to handle transient errors.

        Args:
            messages: Messages (chat history) to send to the model.
            stream: Whether to stream the response or not.
            max_tokens: Maximum number of tokens to generate. Defaults to None (generates until stop sequence or until hitting max context size).
            model_params: Model parameters to use for this request. Defaults to None (uses the default model parameters).
                          see: https://docs.endpoints.anyscale.com/guides/migrate-from-openai#check-any-parameters-to-the-create-call-that-you-might-need-to-change
        Returns:
            ChatResponse or StreamingChatChunk

        Usage:
            >>> from canopy.llm import AnyscaleLLM
            >>> from canopy.models.data_models import UserMessage
            >>> llm = AnyscaleLLM(api_key="secret_YOUR_ANYSCALE_TOKEN",
            >>>                   model_name="meta-llama/Llama-2-70b-chat-hf")
            >>> messages = [UserMessage(content="Hello! How are you?")]
            >>> result = llm.chat_completion(messages)
            >>> print(result.choices[0].message.content)
            "I'm good, how are you?"
        """  # noqa: E501

        model_params_dict: Dict[str, Any] = {}
        model_params_dict.update(
            **self.default_model_params.dict(exclude_defaults=True)
        )
        if model_params:
            model_params_dict.update(**model_params.dict(exclude_defaults=True))

        messages = [m.dict() for m in messages]
        if is_openai_v1():
            #raise NotImplementedError()
            client = openai.OpenAI(
                base_url = ANYSCALE_API_BASE,
                api_key = self.api_key
            )
            response = client.chat.completions.create(
                messages=messages,
                stream=stream,
                max_tokens=max_tokens,
                **model_params_dict
            )
        else:
            response = openai.ChatCompletion.create(
                api_base=ANYSCALE_API_BASE,
                api_key=self.api_key,
                model=self.model_name,
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

    @retry(
        reraise=True,
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(
            OPEN_AI_TRANSIENT_EXCEPTIONS + (json.decoder.JSONDecodeError,
                                            jsonschema.ValidationError)
        ),
    )

    def enforced_function_call(self,
                           messages: Messages,
                           function: Function,
                           *,
                           max_tokens: Optional[int] = None,
                           model_params: Optional[ModelParams] = None) -> dict:
        raise NotImplementedError()


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
