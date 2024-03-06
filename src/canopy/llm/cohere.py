import time
from copy import deepcopy
from typing import Union, Iterable, Optional, Any, Dict, List

from tenacity import retry, stop_after_attempt

try:
    import cohere
except (OSError, ImportError, ModuleNotFoundError):
    _cohere_installed = False
else:
    _cohere_installed = True

from canopy.llm import BaseLLM
from canopy.llm.models import Function
from canopy.models.api_models import (
    _Choice,
    _StreamChoice,
    ChatResponse,
    StreamingChatChunk,
    TokenCounts,
)
from canopy.models.data_models import Context, MessageBase, Messages, Role, Query
from canopy.context_engine.context_builder.stuffing import StuffingContextContent


COMMON_PARAMS = {
    "model",
    "frequency_penalty",
    "logit_bias",
    "max_tokens",
    "presence_penalty",
    "stream",
    "temperature",
}


EQUIVALENT_PARAMS = {
    "top_p": "p",
    "user": "user_name",
}


class CohereLLM(BaseLLM):
    """
    Cohere LLM wrapper built on top of the Cohere Python client.

    Note: Cohere requires a valid API key to use this class.
          You can set the "CO_API_KEY" environment variable to your API key.
    """
    def __init__(self,
                 model_name: str = "command",
                 *,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 ignore_unrecognized_params: Optional[bool] = False,
                 **kwargs: Any,
                 ):
        """
        Initialize the Cohere LLM.

        Args:
            model_name: The name of the model to use. See https://docs.cohere.com/docs/models
            api_key: Your Cohere API key. Defaults to None (uses the "CO_API_KEY" environment variable).
            base_url: The base URL to use for the Cohere API. Defaults to None (uses the "CO_API_URL" environment variable if set, otherwise use default Cohere API URL).
            ignore_unrecognized_params: Flag to suppress errors when unrecognized model params (from other LLMs) are passed to Cohere.
            **kwargs: Generation default parameters to use for each request. See https://platform.openai.com/docs/api-reference/chat/create
                    For example, you can set the temperature, p, etc
                    These params can be overridden by passing a `model_params` argument to the `chat_completion` methods.
        """  # noqa: E501
        super().__init__(model_name)

        if not _cohere_installed:
            raise ImportError(
                "Failed to import cohere. Make sure you install cohere extra "
                "dependencies by running: "
                "pip install canopy-sdk[cohere]"
            )

        try:
            self._client = cohere.Client(api_key, api_url=base_url)
        except cohere.error.CohereError as e:
            raise RuntimeError(
                "Failed to connect to Cohere, please make sure that the CO_API_KEY "
                "environment variable is set correctly.\n"
                f"Error: {e.message}"
            )

        self.ignore_unrecognized_params = ignore_unrecognized_params
        self.default_model_params = kwargs

    def chat_completion(self,
                        system_prompt: str,
                        chat_history: Messages,
                        context: Optional[Context] = None,
                        *,
                        stream: bool = False,
                        max_tokens: Optional[int] = None,
                        model_params: Optional[dict] = None,
                        ) -> Union[ChatResponse, Iterable[StreamingChatChunk]]:
        """
        Chat completion using the Cohere API.

        Note: this function is wrapped in a retry decorator to handle transient errors.

        Args:
            system_prompt: The system prompt to use for the chat completion (preamble).
            chat_history: Messages (chat history) to send to the model.
            context: Knowledge base context to use for the chat completion. Defaults to None (no context).
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
            >>> llm = CohereLLM()
            >>> messages = [UserMessage(content="Hello! How are you?")]
            >>> result = llm.chat_completion(messages)
            >>> print(result.choices[0].message.content)
            "I'm good, how are you?"
        """  # noqa: E501
        model_params_dict: Dict[str, Any] = deepcopy(self.default_model_params)
        model_params_dict.update(
            model_params or {}
        )
        model_params_dict["max_tokens"] = max_tokens

        model_params_dict = self._convert_model_params(model_params_dict)

        connectors = model_params_dict.pop('connectors', None)
        messages: List[Dict[str, Any]] = self._map_messages(chat_history)
        model_name = model_params_dict.pop('model', None) or self.model_name

        if not messages:
            raise RuntimeError("No message provided")

        if system_prompt:
            messages = self._prepend_system_prompt_to_messages(system_prompt, messages)

        try:
            response = self._client.chat(
                model=model_name,
                message=messages.pop()['message'],
                chat_history=messages,
                documents=self.generate_documents_from_context(context),
                stream=stream,
                connectors=[
                    {"id": connector} for connector in connectors
                ] if connectors else None,
                **model_params_dict
            )
        except cohere.error.CohereAPIError as e:
            raise RuntimeError(
                f"Failed to use Cohere's {model_name} model for chat "
                f"completion. "
                f"Underlying Error:\n{e.message}"
            )

        def streaming_iterator(res):
            for chunk in res:
                if chunk.event_type != "text-generation":
                    continue

                choice = _StreamChoice(
                    index=0,
                    delta={
                        "content": chunk.text,
                        "function_call": None,
                        "role": Role.ASSISTANT,
                        "tool_calls": None
                    },
                    finish_reason=None,
                )
                streaming_chat_chunk = StreamingChatChunk(
                    id='',
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=self.model_name,
                    choices=[choice],
                )
                streaming_chat_chunk.id = chunk.id

                yield streaming_chat_chunk

        if stream:
            return streaming_iterator(response)

        return ChatResponse(
            id=response.id,
            created=int(time.time()),
            choices=[_Choice(
                index=0,
                message=MessageBase(
                    role=Role.ASSISTANT,
                    content=response.text,
                ),
                finish_reason="stop",
            )],
            object="chat.completion",
            model=self.model_name,
            usage=TokenCounts(
                prompt_tokens=response.token_count["prompt_tokens"],
                completion_tokens=response.token_count["response_tokens"],
                total_tokens=response.token_count["billed_tokens"],
            ),
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
    )
    def generate_search_queries(self, messages):
        messages = self._map_messages(messages)
        response = self._client.chat(
            model=self.model_name,
            message=messages[-1]['message'],
            chat_history=messages[:-1],
            stream=False,
            search_queries_only=True,
        )
        return [search_query['text'] for search_query in response.search_queries]

    def enforced_function_call(self,
                               system_prompt: str,
                               chat_history: Messages,
                               function: Function,
                               *,
                               max_tokens: Optional[int] = None,
                               model_params: Optional[dict] = None
                               ) -> dict:
        raise NotImplementedError("Cohere LLM doesn't support function calling")

    async def aenforced_function_call(self,
                                      system_prompt: str,
                                      chat_history: Messages,
                                      function: Function, *,
                                      max_tokens: Optional[int] = None,
                                      model_params: Optional[dict] = None):
        raise NotImplementedError("Cohere LLM doesn't support function calling")

    async def achat_completion(self,
                               system_prompt: str,
                               chat_history: Messages,
                               context: Optional[Context] = None,
                               *,
                               stream: bool = False,
                               max_generated_tokens: Optional[int] = None,
                               model_params: Optional[dict] = None,
                               ) -> Union[ChatResponse,
                                          Iterable[StreamingChatChunk]]:
        raise NotImplementedError("Cohere LLM doesn't support async chat completion")

    async def agenerate_queries(self,
                                messages: Messages,
                                *,
                                max_generated_tokens: Optional[int] = None,
                                model_params: Optional[dict] = None,
                                ) -> List[Query]:
        raise NotImplementedError("Cohere LLM doesn't support async query generation")

    def _convert_model_params(self, openai_model_params: dict) -> dict:
        """
        Convert Open AI model params to Cohere equivalents.

        Args:
            openai_model_params: model params passed from client to Canopy API in OpenAI format.

        Returns:
            Model params used with Cohere Chat API.
        """  # noqa: E501
        converted_model_params = {}

        for param in list(openai_model_params.keys()):
            if param in COMMON_PARAMS:
                converted_model_params[param] = openai_model_params.pop(param)
            elif param in EQUIVALENT_PARAMS:
                converted_model_params[EQUIVALENT_PARAMS[param]] = \
                    openai_model_params.pop(param)

        # Scale is -2.0 to 2.0 with OpenAI, but -1.0 to 1.0 with Cohere.
        if presence_penalty := converted_model_params.get("presence_penalty"):
            converted_model_params = presence_penalty * 0.5

        unrecognized_keys = set(openai_model_params.keys())
        default_keys = set(self.default_model_params.keys())

        if unrecognized_keys.difference(default_keys) \
                and not self.ignore_unrecognized_params:
            raise NotImplementedError(
                f"{','.join(unrecognized_keys)} not supported by Cohere Chat API."
            )

        return converted_model_params

    def _map_messages(self, messages: Messages) -> List[dict[str, Any]]:
        """
        Map the messages to format expected by Cohere.

        Cohere Chat API expects message history to be in the format:
        {
          "role": "USER",
          "message": "message text"
        }

        System messages will be passed as user messages.

        Args:
            messages: (chat history) to send to the model.

        Returns:
            list A List of dicts in format expected by Cohere chat API.
        """
        mapped_messages = []

        for message in messages:
            if not message.content:
                continue

            mapped_messages.append({
                "role": "CHATBOT" if message.role == Role.ASSISTANT else "USER",
                "message": message.content,
            })

        return mapped_messages

    def _prepend_system_prompt_to_messages(self,
                                           system_prompt: str,
                                           messages: List[dict[str, Any]]) -> (
                                                List)[dict[str, Any]]:
        """
        Prepend the value passed as the system prompt to the messages.

        Cohere does not have a direct equivalent to the system prompt, and when passing
        documents it's preferred to send the system prompt as the first message instead.
        """
        first_message = messages[0]

        if (first_message["message"] == system_prompt
                and first_message["role"] == "USER"):
            return messages

        system_prompt_messages = [
            {
                "role": "USER",
                "message": system_prompt,
            },
            {
                "role": "CHATBOT",
                "message": "Ok."
            }
        ]

        return system_prompt_messages + messages

    def generate_documents_from_context(
            self, context: Optional[Context]) -> List[Dict[str, Any]]:
        """
        Generate document data to pass to Cohere Chat API from provided context data.

        Args:
            context: Knowledge base context to use for the chat completion.

        Returns:
            documents: list of document objects for Cohere API.
        """
        if not context:
            return []

        if isinstance(context.content, StuffingContextContent):
            return (
                self.generate_documents_from_stuffing_context_content(context.content)
            )

        raise NotImplementedError(
            "Cohere LLM is currently supported only with StuffingContextBuilder."
        )

    def generate_documents_from_stuffing_context_content(
            self,
            content: StuffingContextContent) -> List[Dict[str, Any]]:
        """
        Generate document data to pass to Cohere Chat API from StuffingContextContent.

        Args:
            content: Stuffing context content from the context.

        Returns:
            documents: list of document objects for Cohere API.
        """
        documents = []

        for result in content.root:
            for snippet in result.snippets:
                documents.append(snippet.model_dump())

        return documents
