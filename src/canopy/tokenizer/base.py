from abc import ABC, abstractmethod
from typing import List

from canopy.models.data_models import Messages
from canopy.utils.config import ConfigurableMixin


class BaseTokenizer(ABC, ConfigurableMixin):

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
