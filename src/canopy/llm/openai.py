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
from canopy.utils.openai_exceptions import OPEN_AI_TRANSIENT_EXCEPTIONS
from canopy.llm import BaseLLM
from canopy.llm.models import Function, ModelParams
from canopy.models.api_models import ChatResponse, StreamingChatChunk
from canopy.models.data_models import Messages, Query


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM wrapper built on top of the OpenAI Python client.

    Note: OpenAI requires a valid API key to use this class.
          You can set the "OPENAI_API_KEY" environment variable to your API key.
          Or you can directly set it as follows:
          >>> import openai
          >>> openai.api_key = "YOUR_API_KEY"
    """
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 *,
                 model_params: Optional[ModelParams] = None,
                 ):
        super().__init__(model_name,
                         model_params=model_params)

    @property
    def available_models(self):
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
                          see: https://platform.openai.com/docs/api-reference/chat/create
        Returns:
            ChatResponse or StreamingChatChunk

        Usage:
            >>> from canopy.llm import OpenAILLM
            >>> from canopy.models.data_models import UserMessage
            >>> llm = OpenAILLM()
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
        """
        This function enforces the model to respond with a specific function call.

        To read more about this feature, see: https://platform.openai.com/docs/guides/gpt/function-calling

        Note: this function is wrapped in a retry decorator to handle transient errors.

        Args:
            messages: Messages (chat history) to send to the model.
            function: Function to call. See canopy.llm.models.Function for more details.
            max_tokens: Maximum number of tokens to generate. Defaults to None (generates until stop sequence or until hitting max context size).
            model_params: Model parameters to use for this request. Defaults to None (uses the default model parameters).
                            see: https://platform.openai.com/docs/api-reference/chat/create

        Returns:
            dict: Function call arguments as a dictionary.

        Usage:
            >>> from canopy.llm import OpenAILLM
            >>> from canopy.llm.models import Function, FunctionParameters, FunctionArrayProperty
            >>> from canopy.models.data_models import UserMessage
            >>> llm = OpenAILLM()
            >>> messages = [UserMessage(content="I was wondering what is the capital of France?")]
            >>> function = Function(
            ...     name="query_knowledgebase",
            ...     description="Query search engine for relevant information",
            ...     parameters=FunctionParameters(
            ...         required_properties=[
            ...             FunctionArrayProperty(
            ...                 name="queries",
            ...                 items_type="string",
            ...                 description='List of queries to send to the search engine.',
            ...             ),
            ...         ]
            ...     )
            ... )
            >>> llm.enforced_function_call(messages, function)
            {'queries': ['capital of France']}
        """  # noqa: E501
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
        arguments = json.loads(result["arguments"])

        jsonschema.validate(instance=arguments, schema=function.parameters.dict())
        return arguments

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
