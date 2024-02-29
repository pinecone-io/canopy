from typing import Optional, Any
import os
from canopy.llm import OpenAILLM
from canopy.llm.models import Function
from canopy.models.data_models import Messages

OCTOAI_BASE_URL = "https://text.octoai.run/v1"


class OctoAILLM(OpenAILLM):
    """
    OctoAI LLM wrapper built on top of the OpenAI Python client.

    Note: OctoAI requires a valid API key to use this class.
          You can set the "OCTOAI_API_KEY" environment variable.
    """

    def __init__(
        self,
        model_name: str = "mistral-7b-instruct-fp16",
        *,
        base_url: Optional[str] = OCTOAI_BASE_URL,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        octoai_api_key = api_key or os.environ.get("OCTOAI_API_KEY")
        if not octoai_api_key:
            raise ValueError(
                "OctoAI API key is required to use OctoAI. "
                "If you haven't done it, please sign up at https://octo.ai"
                "The key can be provided as an argument or via the OCTOAI_API_KEY environment variable."
            )
        octoai_base_url = base_url
        super().__init__(model_name, api_key=octoai_api_key, base_url=octoai_base_url, **kwargs)

    def enforced_function_call(
        self,
        system_prompt: str,
        chat_history: Messages,
        function: Function,
        *,
        max_tokens: Optional[int] = None,
        model_params: Optional[dict] = None,
    ) -> dict:
        raise NotImplementedError("OctoAI doesn't support function calling.")

    def aenforced_function_call(self,
                                system_prompt: str,
                                chat_history: Messages,
                                function: Function,
                                *,
                                max_tokens: Optional[int] = None,
                                model_params: Optional[dict] = None
                                ):
        raise NotImplementedError("OctoAI doesn't support function calling.")
