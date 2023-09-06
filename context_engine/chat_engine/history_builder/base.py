from abc import ABC, abstractmethod
from typing import Tuple

from context_engine.knoweldge_base.tokenizer.base import Tokenizer
from context_engine.models.data_models import Messages


class BaseHistoryBuilder(ABC):

    def __init__(self,
                 tokenizer: Tokenizer,
                 min_history_messages: int = 1):
        self._tokenizer = tokenizer
        self._min_history_messages = min_history_messages

    @abstractmethod
    def build(self,
              full_history: Messages,
              max_tokens: int) -> Tuple[Messages, int]:
        raise NotImplementedError

    async def abuild(self,
                     full_history: Messages,
                     max_tokens: int) -> Tuple[Messages, int]:
        raise NotImplementedError()
