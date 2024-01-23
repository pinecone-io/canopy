import cohere

from typing import List, Optional
from .base import BaseTokenizer
from ..models.data_models import Messages


class CohereAPITokenizer(BaseTokenizer):
    """
    Tokenizer for Cohere models, based on the Cohere Tokenize API.

    Usage:
    Initialize the singleton tokenizer with the CohereAPITokenizer class:
    >>> from canopy.tokenizer import Tokenizer
    >>> Tokenizer.initialize(tokenizer_class=CohereAPITokenizer, model_name="embed-multilingual-v3.0")

    You can then use the tokenizer instance from anywhere in the code:
    >>> from canopy.tokenizer import Tokenizer
    >>> tokenizer = Tokenizer()
    >>> tokenizer.tokenize("Hello world!")
    ['Hello', ' world', '!']
    """  # noqa: E501

    MESSAGE_TOKENS_OVERHEAD = 3
    FIXED_PREFIX_TOKENS = 3

    def __init__(self,
                 model_name: Optional[str] = None,
                 *,
                 api_key: Optional[str] = None,
                 api_url: Optional[str] = None):
        """
        Initialize the tokenizer.

        Args:
            model_name: The name of the model to use.
            api_key: Your Cohere API key. Defaults to None (uses the "CO_API_KEY" environment variable).
            api_url: The base URL to use for the Cohere API. Defaults to None (uses the "CO_API_URL" environment variable if set, otherwise use default Cohere API URL).
        """  # noqa: E501
        self.model_name = model_name
        self._client = cohere.Client(api_key, api_url=api_url)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text using Cohere Tokenize API.

        Args:
            text: The text to tokenize.

        Returns:
            The list of tokens.
        """
        if not text:
            return []

        tokens = self._client.tokenize(text, model=self.model_name)
        return tokens.token_strings

    def detokenize(self, tokens: List[str]) -> str:
        """
        Detokenize a list of tokens that were previously tokenized using this tokenizer.

        Args:
            tokens: The list of tokens to detokenize.

        Returns:
            The detokenized text as a string.
        """
        if not isinstance(tokens, List):
            raise TypeError(f"detokenize expects List[str], got f{type(tokens)}")
        return "".join(tokens)

    def messages_token_count(self, messages: Messages) -> int:
        """
        Count the number of tokens in a list of messages as expected to be counted by Cohere models.
        Account for the overhead of the messages structure.
        Taken from: https://github.com/openai/openai-cookbook/.../How_to_format_inputs_to_ChatGPT_models.ipynb

        Args:
            messages: The list of messages to count the tokens of.

        Returns:
            The number of tokens in the messages, as expected to be counted by OpenAI models.
        """  # noqa: E501
        num_tokens = 0
        for message in messages:
            num_tokens += self.MESSAGE_TOKENS_OVERHEAD
            for key, value in message.dict().items():
                num_tokens += self.token_count(value)
        num_tokens += self.FIXED_PREFIX_TOKENS
        return num_tokens
