import tiktoken
from typing import List
from .base import BaseTokenizer
from ..models.data_models import Messages


class OpenAITokenizer(BaseTokenizer):
    """
    Tokenizer for OpenAI models, based on the tiktoken library.

    Usage:
    Initialize the singleton tokenizer with the OpenAITokenizer class:
    >>> from canopy.tokenizer import Tokenizer
    >>> Tokenizer.initialize(tokenizer_class=OpenAITokenizer, model_name="gpt-3.5-turbo")

    You can then use the tokenizer instance from anywhere in the code:
    >>> from canopy.tokenizer import Tokenizer
    >>> tokenizer = Tokenizer()
    >>> tokenizer.tokenize("Hello world!")
    ['Hello', ' world', '!']
    """  # noqa: E501

    MESSAGE_TOKENS_OVERHEAD = 3
    FIXED_PREFIX_TOKENS = 3

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the tokenizer.

        Args:
            model_name: The name of the model to use. Defaults to "gpt-3.5-turbo".
                        You can find the list of available models here: https://github.com/openai/tiktoken/blob/39f29cecdb6fc38d9a3434e5dd15e4de58cf3c80/tiktoken/model.py#L19C1-L19C18
                        As you can see, both gpt-3.5 and gpt-4 are using the same cl100k_base tokenizer.
        """  # noqa: E501
        self._encoder = tiktoken.encoding_for_model(model_name)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text using tiktoken.

        Args:
            text: The text to tokenize.

        Returns:
            The list of tokens.
        """
        return [self._encoder.decode([encoded_token])
                for encoded_token in self._encode(text)]

    def detokenize(self, tokens: List[str]) -> str:
        """
        Detokenize a list of tokens that were previously tokenized using this tokenizer.

        Args:
            tokens: The list of tokens to detokenize.

        Returns:
            The detokenized text as a string.
        """
        if not isinstance(tokens, List):
            raise TypeError(f"detokenize expect List[str], got f{type(tokens)}")
        return "".join(tokens)

    def token_count(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: The text to count the tokens of.

        Returns:
            The number of tokens in the text.
        """
        return len(self._encode(text))

    def _encode(self, text):
        return self._encoder.encode(text, disallowed_special=())

    def messages_token_count(self, messages: Messages) -> int:
        """
        Count the number of tokens in a list of messages as expected to be counted by OpenAI models.
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
