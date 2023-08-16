from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def detokenize(self, tokens: List[str]) -> str:
        pass

    def token_count(self, text: str) -> int:
        return len(self.tokenize(text))
