import json
from copy import deepcopy
from typing import Optional, Any, Dict, cast, Union, Iterable, List

import jsonschema
import openai
from openai.types.chat import ChatCompletionToolParam
from tenacity import retry, stop_after_attempt, retry_if_exception_type

import canopy
from canopy.llm import OpenAILLM
from canopy.llm.models import Function, FunctionParameters, FunctionArrayProperty
from canopy.models.api_models import ChatResponse, StreamingChatChunk
from canopy.models.data_models import Messages, UserMessage, Query


class AzureOpenAILLM(OpenAILLM):
    """
    Azure OpenAI LLM wrapper built on top of the OpenAI Python client.

    Note: Azure OpenAI services require a valid API key, and an Azure endpoint
          "AZURE_OPENAI_KEY" environment variable to your API key.
    Note: If you want to pass an OpenAI organization, you need to set an environment variable "OPENAI_ORG_ID". Note
          that this name is different from the environment variable name for passing an organization to the parent
          class (OpenAILLM), which is "OPENAI_ORG".
    """
    def __init__(self,
                 model_name: Optional[str] = None,
                 *,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs: Any,
                 ):
        """
        Initialize the Azure OpenAI LLM.

        Args:
            model_name: The name of the model to use. See https://platform.openai.com/docs/models
            api_key: Your Azure OpenAI API key. Defaults to None (uses the "OPENAI_API_KEY" environment variable).
            base_url: The base URL to use for the Azure OpenAI API. Will use the AZURE_OPENAI_ENDPOINT environment variable if set.
            **kwargs: Generation default parameters to use for each request. See https://platform.openai.com/docs/api-reference/chat/create
                    For example, you can set the temperature, top_p etc
                    These params can be overridden by passing a `model_params` argument to the `chat_completion` or `enforced_function_call` methods.

        >>> import os
        >>> os.environ['OPENAI_API_VERSION'] = "OPENAI API VERSION (FOUND IN AZURE RESTAPI DOCS)"
        >>> os.environ['AZURE_OPENAI_ENDPOINT'] = "AZURE ENDPOINT"
        >>> os.environ['AZURE_OPENAI_API_KEY'] = "YOUR KEY"
        >>> os.environ['AZURE_DEPLOYMENT'] = "YOUR AZURE DEPLOYMENT'S NAME"


        >>> from canopy.llm.azure_openai_llm import AzureOpenAILLM
        >>> from canopy.models.data_models import UserMessage
        >>> llm = AzureOpenAILLM()
        >>> messages = [UserMessage(content="Hello! How are you?")]
        >>> llm.chat_completion(messages)

        """  # noqa: E501
        super().__init__(model_name)

        if not environ.get('AZURE_OPENAI_API_KEY'):
            raise EnvironmentError('Please set your Azure OpenAI API key environment variable ('
                                   'export AZURE_OPENAI_API_KEY=<your azure openai api key>). See here for more '
                                   'information: '
                                   'https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Cpython&pivots=programming-language-python')

        if not environ.get('AZURE_OPENAI_ENDPOINT'):
            raise EnvironmentError("Please set your Azure OpenAI endpoint environment variable ('export "
                                   "AZURE_OPENAI_ENDPOINT=<your endpoint>'). See here for more information "
                                   "https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Cpython&pivots=programming-language-python")

        if not environ.get('OPENAI_API_VERSION'):
            raise EnvironmentError("Please set your Azure OpenAI API version. ('export OPENAI_API_VERSION=<your API "
                                   "version"
                                   ">'). See here for more information "
                                   "https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Cpython&pivots=programming-language-python")

        if azure_deployment is None:
            if not environ.get('AZURE_DEPLOYMENT'):
                raise EnvironmentError('You need to set an environment variable for AZURE_DEPLOYMENT to the name of '
                                       'your Azure deployment')
            azure_deployment = os.getenv('AZURE_DEPLOYMENT')



        self._client = openai.AzureOpenAI(
            api_key=api_key,
            api_version="2023-10-01-preview",
            azure_endpoint=base_url,
        )
        self.default_model_params = kwargs

    @property
    def available_models(self):
        raise NotImplementedError()

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
            >>> from canopy.llm import AzureOpenAILLM
            >>> from canopy.llm.models import Function, FunctionParameters, FunctionArrayProperty
            >>> from canopy.models.data_models import UserMessage
            >>> llm = AzureOpenAILLM()
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
        model_params_dict.update(model_params or {})

        function_dict = cast(ChatCompletionToolParam, function.dict())

        chat_completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=[m.dict() for m in messages],
            functions=[function_dict],
            function_call={"name": function.name},
            max_tokens=max_tokens,
            **model_params_dict,
        )

        result = chat_completion.choices[0].message.function_call
        arguments = json.loads(result.arguments)

        jsonschema.validate(instance=arguments, schema=function.parameters.dict())
        return arguments

    async def achat_completion(
        self,
        messages: Messages,
        *,
        stream: bool = False,
        max_generated_tokens: Optional[int] = None,
        model_params: Optional[dict] = None,
    ) -> Union[ChatResponse, Iterable[StreamingChatChunk]]:
        raise NotImplementedError()

    async def agenerate_queries(
        self,
        messages: Messages,
        *,
        max_generated_tokens: Optional[int] = None,
        model_params: Optional[dict] = None,
    ) -> List[Query]:
        raise NotImplementedError()
