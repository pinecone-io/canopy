from abc import ABC, abstractmethod
from context_engine.models.data_models import Messages


class BaseHistoryBuilder(ABC):

    @abstractmethod
    def build(self, messages: Messages, max_tokens: int) -> Messages:
        pass
