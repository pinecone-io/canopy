from typing import List, Optional, Type

from .openai import OpenAITokenizer
from .base import BaseTokenizer
from ..models.data_models import Messages


class Tokenizer:

    """
    Singleton class for tokenization.
    The singleton behavior unify tokenization across the system.

    Usage:

    To initialize the tokenizer, call Tokenizer.initialize(tokenizer_class, *args, **kwargs)
    >>> from canopy.tokenizer import Tokenizer
    >>> Tokenizer.initialize()

    Then, you can instantiate a tokenizer instance by calling Tokenizer() from anywhere in the code and use it:
    >>> tokenizer = Tokenizer()
    >>> tokenizer.tokenize("Hello world!")
    ['Hello', 'world', '!']
    >>> tokenizer.detokenize(['Hello', 'world', '!'])
    'Hello world!'
    """  # noqa: E501

    _instance = None
    _tokenizer_instance: Optional[BaseTokenizer] = None
    _initialized = False

    DEFAULT_TOKENIZER_CLASS = OpenAITokenizer

    def __new__(cls):
        if not cls._initialized:
            raise ValueError("Tokenizer must be initialized using "
                             "Tokenizer.initialize(tokenizer_class, *args, **kwargs)")
        if not cls._instance:
            cls._instance = super(Tokenizer, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls,
                   tokenizer_class: Type[BaseTokenizer] = DEFAULT_TOKENIZER_CLASS,
                   **kwargs):
        """
        Initialize the tokenizer singleton.

        Args:
            tokenizer_class: The tokenizer class to use. Must be a subclass of BaseTokenizer. Defaults to OpenAITokenizer.
            **kwargs: Keyword arguments to pass to the underlying `Tokenizer` class constructor.

        Examples:
            Initialize the tokenizer with the default tokenizer class:

            >>> from canopy.tokenizer import Tokenizer
            >>> Tokenizer.initialize()

            Initialize the tokenizer with a custom tokenizer class:

            >>> from canopy.tokenizer import Tokenizer
            >>> from canopy.tokenizer.base import BaseTokenizer
            >>> class MyTokenizer(BaseTokenizer):
            ...     def tokenize(self, text: str) -> List[str]:
            ...         return text.split()
            ...     def detokenize(self, tokens: List[str]) -> str:
            ...         return " ".join(tokens)
            ...     def messages_token_count(self, messages) -> int:
            ...         return sum([self.token_count(message) + 3 for message in messages])
            >>> Tokenizer.initialize(MyTokenizer)

            Then, you can instantiate a tokenizer instance by calling Tokenizer() from anywhere in the code:

            >>> from canopy.tokenizer import Tokenizer
            >>> tokenizer = Tokenizer()
        """  # noqa: E501
        if not issubclass(tokenizer_class, BaseTokenizer):
            raise ValueError("Invalid tokenizer class provided")
        if issubclass(tokenizer_class, Tokenizer):
            raise ValueError("Tokenizer singleton cannot be passed as tokenizer_class")
        cls._tokenizer_instance = tokenizer_class(**kwargs)
        cls._initialized = True

    @classmethod
    def clear(cls):
        """
        Clear the tokenizer singleton.
        """
        cls._instance = None
        cls._tokenizer_instance = None
        cls._initialized = False

    @classmethod
    def initialize_from_config(cls, config: dict):
        """
        Initialize the tokenizer singleton from a config dictionary.
        Used by the config module to initialize the tokenizer from a config file.

        Args:
            config: A dictionary containing the tokenizer configuration. If not provided, the OpenAITokenizer will be used.

        Usage:
            >>> from canopy.tokenizer import Tokenizer
            >>> config = {
            ...     "type": "OpenAITokenizer",
            ...     "model_name": "gpt2"
            ... }
            >>> Tokenizer.initialize_from_config(config)
        """  # noqa: E501
        if cls._initialized:
            raise ValueError("Tokenizer has already been initialized")
        config["type"] = config.get("type", cls.DEFAULT_TOKENIZER_CLASS.__name__)
        cls._tokenizer_instance = BaseTokenizer.from_config(config)
        cls._initialized = True

    def tokenize(self, text: str) -> List[str]:
        """
        Splits a text into tokens.

        Args:
            text: The text to tokenize as a string.

        Returns:
            A list of tokens.
        """
        return self._tokenizer_instance.tokenize(text)  # type: ignore[union-attr]

    def detokenize(self, tokens: List[str]) -> str:
        """
        Joins a list of tokens into a text.

        Args:
            tokens: The tokens to join as a list of strings. Consider using tokenize() first.

        Returns:
            The joined text as a string.
        """  # noqa: E501
        return self._tokenizer_instance.detokenize(tokens)   # type: ignore[union-attr]

    def token_count(self, text: str) -> int:
        """
        Counts the number of tokens in a text.

        Args:
            text: The text to count as a string.

        Returns:
            The number of tokens in the text.
        """
        return self._tokenizer_instance.token_count(text)   # type: ignore[union-attr]

    def messages_token_count(self, messages: Messages) -> int:
        """
        Counts the number of tokens in a Messages object.
        Behind the scenes, for each LLM provider there might be a different overhead for each message in the prompt,
        which is not necessarily the same as the number of tokens in the message text.
        This method takes care of that overhead and returns the total number of tokens in the prompt, as counted by the LLM provider.

        Args:
            messages: The Messages object to count.

        Returns:
            The number of tokens in the Messages object.
        """  # noqa: E501
        return self._tokenizer_instance.messages_token_count(messages)   # type: ignore[union-attr] # noqa: E501
