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

    Azure OpenAI services require a valid API key, and an Azure endpoint. You will need
    To set the following environment variables:
    - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key.
    - AZURE_OPENAI_ENDPOINT: Your Azure endpoint, including the resource, e.g. `https://example-resource.azure.openai.com/`
    
    To use a specific custom model deployment, simply pass the deployment name to the `model_name` argument
    
    Note: If you want to set an OpenAI organization, you would need to set environment variable "OPENAI_ORG_ID". 
          This is different from the environment variable for passing an organization to the parent class (OpenAILLM), which is "OPENAI_ORG".
    """  # noqa: E501
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 *,
                 api_key: Optional[str] = None,
                 api_version: str = "2023-12-01-preview",
                 azure_endpoint: Optional[str] = None,
                 organization: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs: Any,
                 ):
        """
        Initialize the Azure OpenAI LLM.

        Args:
            model_name: Either your custom model deployment, or the name of a public OpenAI model to use 
            api_key: Your Azure OpenAI API key. Defaults to None (uses the "AZURE_OPENAI_API_KEY" environment variable).
            base_url: The base URL to use for the Azure OpenAI API. Will use the AZURE_OPENAI_ENDPOINT environment variable if set.
            **kwargs: Generation default parameters to use for each request. See https://platform.openai.com/docs/api-reference/chat/create
                    For example, you can set the temperature, top_p etc
                    These params can be overridden by passing a `model_params` argument to the `chat_completion` or `enforced_function_call` methods.


        >>> from canopy.llm.azure_openai_llm import AzureOpenAILLM
        >>> from canopy.models.data_models import UserMessage
        >>> llm = AzureOpenAILLM()
        >>> messages = [UserMessage(content="Hello! How are you?")]
        >>> llm.chat_completion(messages)

        """  # noqa: E501
        self.model_name = model_name

        # if not environ.get('AZURE_OPENAI_API_KEY'):
        #     raise EnvironmentError('Please set your Azure OpenAI API key environment variable ('
        #                            'export AZURE_OPENAI_API_KEY=<your azure openai api key>). See here for more '
        #                            'information: '
        #                            'https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Cpython&pivots=programming-language-python')
        #
        # if not environ.get('OPENAI_API_VERSION'):
        #     raise EnvironmentError("Please set your Azure OpenAI API version. ('export OPENAI_API_VERSION=<your API "
        #                            "version"
        #                            ">'). See here for more information "
        #                            "https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Cpython&pivots=programming-language-python")
        #
        # if azure_deployment is None:
        #     if not environ.get('AZURE_DEPLOYMENT'):
        #         raise EnvironmentError('You need to set an environment variable for AZURE_DEPLOYMENT to the name of '
        #                                'your Azure deployment')
        #     azure_deployment = os.getenv('AZURE_DEPLOYMENT')

        self._client = openai.AzureOpenAI(
            azure_deployment=model_name,
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            organization=organization,
            base_url=base_url,
        )
        self.default_model_params = kwargs

    @property
    def available_models(self):
        raise NotImplementedError("Azure OpenAI LLM does not support listing available models")

    # @retry(
    #     reraise=True,
    #     stop=stop_after_attempt(3),
    #     retry=retry_if_exception_type(
    #         (json.decoder.JSONDecodeError,
    #          jsonschema.ValidationError)
    #     ),
    # )
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
