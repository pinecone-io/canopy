from typing import Optional, Any
import os
from canopy.llm import OpenAILLM
from canopy.llm.models import Function
from canopy.models.data_models import Messages

ANYSCALE_BASE_URL = "https://api.endpoints.anyscale.com/v1"


class AnyscaleLLM(OpenAILLM):
    """
    Anyscale LLM wrapper built on top of the OpenAI Python client.

    Note: Anyscale requires a valid API key to use this class.
          You can set the "ANYSCALE_API_KEY" environment variable.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        *,
        base_url: Optional[str] = ANYSCALE_BASE_URL,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        ae_api_key = api_key or os.environ.get("ANYSCALE_API_KEY")
        if not ae_api_key:
            raise ValueError(
                "Anyscale API key is required to use Anyscale. "
                "Please provide it as an argument "
                "or set the ANYSCALE_API_KEY environment variable."
            )
        ae_base_url = base_url
        super().__init__(model_name, api_key=ae_api_key, base_url=ae_base_url, **kwargs)

    def enforced_function_call(
        self,
        system_prompt: str,
        chat_history: Messages,
        function: Function,
        *,
        max_tokens: Optional[int] = None,
        model_params: Optional[dict] = None,
    ) -> dict:
        raise NotImplementedError()

    def aenforced_function_call(self,
                                system_prompt: str,
                                chat_history: Messages,
                                function: Function,
                                *,
                                max_tokens: Optional[int] = None,
                                model_params: Optional[dict] = None
                                ):
        raise NotImplementedError()
