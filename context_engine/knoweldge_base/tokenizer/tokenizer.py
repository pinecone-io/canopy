from abc import ABC, abstractmethod
from typing import List, Optional

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


class Tokenizer(BaseTokenizer):
    _instance = None
    _tokenizer_instance: Optional[BaseTokenizer] = None
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
        if issubclass(tokenizer_class, Tokenizer):
            raise ValueError("Tokenizer singleton cannot be passed as tokenizer_class")
        cls._tokenizer_instance = tokenizer_class(*args, **kwargs)
        cls._initialized = True

    @classmethod
    def clear(cls):
        cls._instance = None
        cls._tokenizer_instance = None
        cls._initialized = False

    def tokenize(self, text: str) -> List[str]:
        return self._tokenizer_instance.tokenize(text)  # type: ignore[union-attr]

    def detokenize(self, tokens: List[str]) -> str:
        return self._tokenizer_instance.detokenize(tokens)   # type: ignore[union-attr]

    def token_count(self, text: str) -> int:
        return self._tokenizer_instance.token_count(text)   # type: ignore[union-attr]

    def messages_token_count(self, messages) -> int:
        return self._tokenizer_instance.messages_token_count(messages)   # type: ignore[union-attr] # noqa: E501
