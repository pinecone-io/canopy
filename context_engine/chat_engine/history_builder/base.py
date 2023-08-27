from abc import ABC, abstractmethod
from typing import Tuple

from context_engine.knoweldge_base.tokenizer.base import Tokenizer
from context_engine.models.data_models import Messages


class BaseHistoryBuilder(ABC):

    def __init__(self,
                 tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    @abstractmethod
    def build(self,
              full_history: Messages,
              max_tokens: int) -> Tuple[Messages, int]:
        raise NotImplementedError

    async def abuild(self,
                     full_history: Messages,
                     max_tokens: int) -> Tuple[Messages, int]:
        raise NotImplementedError()
