from typing import List, Optional

from .openai import OpenAITokenizer
from .base import BaseTokenizer


class Tokenizer(BaseTokenizer):
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
    def initialize(cls, tokenizer_class=DEFAULT_TOKENIZER_CLASS, *args, **kwargs):
        if cls._initialized:
            raise ValueError("Tokenizer has already been initialized")
        if not issubclass(tokenizer_class, BaseTokenizer):
            raise ValueError("Invalid tokenizer class provided")
        if issubclass(tokenizer_class, Tokenizer):
            raise ValueError("Tokenizer singleton cannot be passed as tokenizer_class")
        cls._tokenizer_instance = tokenizer_class(*args, **kwargs)
        cls._initialized = True

    @classmethod
    def clear(cls):
        cls._instance = None
        cls._tokenizer_instance = None
        cls._initialized = False

    @classmethod
    def initialize_from_config(cls, config: dict):
        return Tokenizer.initialize(cls,
                                    tokenizer_class=config.get("type"),
                                    **config.get("params", {}))

    def tokenize(self, text: str) -> List[str]:
        return self._tokenizer_instance.tokenize(text)  # type: ignore[union-attr]

    def detokenize(self, tokens: List[str]) -> str:
        return self._tokenizer_instance.detokenize(tokens)   # type: ignore[union-attr]

    def token_count(self, text: str) -> int:
        return self._tokenizer_instance.token_count(text)   # type: ignore[union-attr]

    def messages_token_count(self, messages) -> int:
        return self._tokenizer_instance.messages_token_count(messages)   # type: ignore[union-attr] # noqa: E501
