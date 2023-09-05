from abc import ABC, abstractmethod
from typing import List

from context_engine.models.data_models import Messages


class BaseTokenizer(ABC):

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def detokenize(self, tokens: List[str]) -> str:
        pass

    def token_count(self, text: str) -> int:
        return len(self.tokenize(text))

    @abstractmethod
    def messages_token_count(self, messages: Messages) -> int:
        pass


class Tokenizer:
    _instance = None
    _tokenizer_instance = None
    _initialized = False

    def __new__(cls):
        if not cls._initialized:
            raise ValueError("Tokenizer must be initialized using "
                             "Tokenizer.initialize(tokenizer_class, *args, **kwargs)")
        if not cls._instance:
            cls._instance = super(Tokenizer, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, tokenizer_class, *args, **kwargs):
        if cls._initialized:
            raise ValueError("Tokenizer has already been initialized")
        if not issubclass(tokenizer_class, BaseTokenizer):
            raise ValueError("Invalid tokenizer class provided")
        cls._tokenizer_instance = tokenizer_class(*args, **kwargs)
        cls._initialized = True

    def tokenize(self, text: str) -> List[str]:
        return self._tokenizer_instance.tokenize(text)

    def detokenize(self, tokens: List[str]) -> str:
        return self._tokenizer_instance.detokenize(tokens)  # type: ignore[attr-defined]

    def token_count(self, text: str) -> int:
        return self._tokenizer_instance.token_count(text)  # type: ignore[attr-defined]

    def messages_token_count(self, messages) -> int:
        return self._tokenizer_instance.messages_token_count(messages)  # type: ignore[attr-defined] # noqa: E501
