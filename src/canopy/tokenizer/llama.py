import tiktoken
from typing import List
from .base import BaseTokenizer
from ..models.data_models import Messages
from tokenizers import Tokenizer


class LlamaTokenizer(BaseTokenizer):
    """
    Tokenizer for Llama models, based on the tokenizers library.

    Usage:
    Initialize the singleton tokenizer with the LlamaTokenizer class:
    >>> from canopy.tokenizer import Tokenizer
    >>> Tokenizer.initialize(tokenizer_class=LlamaTokenizer, 
                             hf_token="hf_YOUR_HUGGINGFACE_TOKEN",
                             model_name="meta-llama/Llama-2-7b-chat-hf")
    You can then use the tokenizer instance from anywhere in the code:
    >>> from canopy.tokenizer import Tokenizer
    >>> tokenizer = Tokenizer()
    >>> tokenizer.tokenize("Hello world!")
    ['<s>', '▁Hello', '▁World', '!']
    """  # noqa: E501

    MESSAGE_TOKENS_OVERHEAD = 3
    FIXED_PREFIX_TOKENS = 3

    def __init__(self, hf_token: str, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize the tokenizer.

        Args:
            hf_token: Huggingface token
            model_name: The name of the model to use. Defaults to "meta-llama/Llama-2-7b-chat-hf".
        """  # noqa: E501
        self._encoder = Tokenizer.from_pretrained(model_name, auth_token=hf_token)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text using tiktoken.

        Args:
            text: The text to tokenize.

        Returns:
            The list of tokens.
        """
        return self._encode(text).tokens

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
        return len(self._encode(text).ids)

    def _encode(self, text):
        # Return Encoding objects, which contains attributes ids and tokens
        return self._encoder.encode(text)

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
