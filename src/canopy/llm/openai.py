from copy import deepcopy
from typing import Union, Iterable, Optional, Any, Dict, List, cast

import jsonschema
import openai
import json

from openai.types.chat import ChatCompletionToolParam
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
)
from canopy.llm import BaseLLM
from canopy.llm.models import Function
from canopy.models.api_models import ChatResponse, StreamingChatChunk
from canopy.models.data_models import Messages, Query


def _format_openai_error(e):
    try:
        return e.response.json()['error']['message']
    except Exception:
        return str(e)


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
                 api_key: Optional[str] = None,
                 organization: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs: Any,
                 ):
        """
        Initialize the OpenAI LLM.

        Args:
            model_name: The name of the model to use. See https://platform.openai.com/docs/models
            api_key: Your OpenAI API key. Defaults to None (uses the "OPENAI_API_KEY" environment variable).
            organization: Your OpenAI organization. Defaults to None (uses the "OPENAI_ORG" environment variable if set, otherwise uses the "default" organization).
            base_url: The base URL to use for the OpenAI API. Defaults to None (uses the default OpenAI API URL).
            **kwargs: Generation default parameters to use for each request. See https://platform.openai.com/docs/api-reference/chat/create
                    For example, you can set the temperature, top_p etc
                    These params can be overridden by passing a `model_params` argument to the `chat_completion` or `enforced_function_call` methods.
        """  # noqa: E501
        super().__init__(model_name)
        try:
            self._client = openai.OpenAI(api_key=api_key,
                                         organization=organization,
                                         base_url=base_url)
        except openai.OpenAIError as e:
            raise RuntimeError(
                "Failed to connect to OpenAI, please make sure that the OPENAI_API_KEY "
                "environment variable is set correctly.\n"
                f"Error: {_format_openai_error(e)}"
            )

        self.default_model_params = kwargs

    @property
    def available_models(self):
        return [k.id for k in self._client.models.list()]

    def chat_completion(self,
                        messages: Messages,
                        *,
                        stream: bool = False,
                        max_tokens: Optional[int] = None,
                        model_params: Optional[dict] = None,
                        ) -> Union[ChatResponse, Iterable[StreamingChatChunk]]:
        """
        Chat completion using the OpenAI API.

        Note: this function is wrapped in a retry decorator to handle transient errors.

        Args:
            messages: Messages (chat history) to send to the model.
            stream: Whether to stream the response or not.
            max_tokens: Maximum number of tokens to generate. Defaults to None (generates until stop sequence or until hitting max context size).
            model_params: Model parameters to use for this request. Defaults to None (uses the default model parameters).
                          Dictonary of parametrs to override the default model parameters if set on initialization.
                          For example, you can pass: {"temperature": 0.9, "top_p": 1.0} to override the default temperature and top_p.
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

        model_params_dict: Dict[str, Any] = deepcopy(self.default_model_params)
        model_params_dict.update(
            model_params or {}
        )

        messages = [m.dict() for m in messages]
        try:
            response = self._client.chat.completions.create(model=self.model_name,
                                                            messages=messages,
                                                            stream=stream,
                                                            max_tokens=max_tokens,
                                                            **model_params_dict)
        except openai.OpenAIError as e:
            provider_name = self.__class__.__name__.replace("LLM", "")
            raise RuntimeError(
                f"Failed to use {provider_name}'s {self.model_name} model for chat "
                f"completion.\n"
                f"Error: {_format_openai_error(e)}"
            )

        def streaming_iterator(response):
            for chunk in response:
                yield StreamingChatChunk.parse_obj(chunk)

        if stream:
            return streaming_iterator(response)

        return ChatResponse.parse_obj(response)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(
            (json.decoder.JSONDecodeError,
             jsonschema.ValidationError)
        ),
    )
    def enforced_function_call(self,
                               messages: Messages,
                               function: Function,
                               *,
                               max_tokens: Optional[int] = None,
                               model_params: Optional[dict] = None,) -> dict:
        """
        This function enforces the model to respond with a specific function call.

        To read more about this feature, see: https://platform.openai.com/docs/guides/gpt/function-calling

        Note: this function is wrapped in a retry decorator to handle transient errors.

        Args:
            messages: Messages (chat history) to send to the model.
            function: Function to call. See canopy.llm.models.Function for more details.
            max_tokens: Maximum number of tokens to generate. Defaults to None (generates until stop sequence or until hitting max context size).
            model_params: Model parameters to use for this request. Defaults to None (uses the default model parameters).
                          Overrides the default model parameters if set on initialization.
                          For example, you can pass: {"temperature": 0.9, "top_p": 1.0} to override the default temperature and top_p.
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

        model_params_dict: Dict[str, Any] = deepcopy(self.default_model_params)
        model_params_dict.update(
            model_params or {}
        )

        function_dict = cast(ChatCompletionToolParam,
                             {"type": "function", "function": function.dict()})

        try:
            chat_completion = self._client.chat.completions.create(
                messages=[m.dict() for m in messages],
                model=self.model_name,
                tools=[function_dict],
                tool_choice={"type": "function",
                             "function": {"name": function.name}},
                max_tokens=max_tokens,
                **model_params_dict
            )
        except openai.OpenAIError as e:
            provider_name = self.__class__.__name__.replace("LLM", "")
            raise RuntimeError(
                f"Failed to use {provider_name}'s {self.model_name} model for "
                f"chat completion with enforced function calling.\n"
                f"Error: {_format_openai_error(e)}"
            )

        result = chat_completion.choices[0].message.tool_calls[0].function.arguments
        arguments = json.loads(result)

        jsonschema.validate(instance=arguments, schema=function.parameters.dict())
        return arguments

    async def achat_completion(self,
                               messages: Messages, *, stream: bool = False,
                               max_generated_tokens: Optional[int] = None,
                               model_params: Optional[dict] = None,
                               ) -> Union[ChatResponse,
                                          Iterable[StreamingChatChunk]]:
        raise NotImplementedError()

    async def agenerate_queries(self,
                                messages: Messages,
                                *,
                                max_generated_tokens: Optional[int] = None,
                                model_params: Optional[dict] = None,
                                ) -> List[Query]:
        raise NotImplementedError()
