from copy import deepcopy
from typing import Union, Iterable, Optional, Any, Dict, List, cast

import jsonschema
import openai
import json, os

from openai.types.chat import ChatCompletionToolParam
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
)
from canopy.llm import OpenAILLM
from canopy.llm.models import Function
from canopy.models.api_models import ChatResponse, StreamingChatChunk
from canopy.models.data_models import Messages, Query

ANYSCALE_BASE_URL = "https://api.endpoints.anyscale.com/v1"
class AnyscaleLLM(OpenAILLM):
    """
    Anyscale LLM wrapper built on top of the OpenAI Python client.

    Note: Anyscale requires a valid API key to use this class.
          You can set the "ANYSCALE_API_KEY" environment variable to your API key.
          Or you can directly set it as follows:
          >>> import openai
          >>> openai.api_key = "YOUR_API_KEY"
    """
    def __init__(self,
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 *,
                 api_key: Optional[str] = None,
                 organization: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs: Any,
                 ):
        ae_api_key = api_key or os.environ.get("ANYSCALE_API_KEY") 
        if not ae_api_key: 
            raise ValueError(f"Anyscale API key is required to use Anyscale. " 
                             f"Please provide it as an argument or set the ANYSCALE_API_KEY environment variable.")
        ae_base_url = base_url or os.environ.get("ANYSCALE_BASE_URL", ANYSCALE_BASE_URL)
        super().__init__(model_name,
                       api_key = ae_api_key,
                       organization = organization,
                       base_url = ae_base_url,
                       **kwargs)

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
        raise NotImplementedError()


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
