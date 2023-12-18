from abc import ABC, abstractmethod
from typing import Tuple, Optional

from canopy.tokenizer import Tokenizer
from canopy.models.data_models import Messages, SystemMessage


class HistoryPruner(ABC):

    def __init__(self):
        self._tokenizer = Tokenizer()

    @abstractmethod
    def build(self,
              chat_history: Messages,
              max_tokens: int,
              system_prompt: Optional[str] = None,
              context: Optional[str] = None,
              ) -> Tuple[Messages, int]:
        raise NotImplementedError

    async def abuild(self,
                     chat_history: Messages,
                     max_tokens: int) -> Tuple[Messages, int]:
        raise NotImplementedError()

    def _max_tokens_history(self,
                            max_tokens: int,
                            system_prompt: Optional[str] = None,
                            context: Optional[str] = None, ) -> int:
        if system_prompt is not None:
            max_tokens -= self._tokenizer.messages_token_count([SystemMessage(content=system_prompt)])

        if context is not None:
            max_tokens -= self._tokenizer.token_count(context)

        return max_tokens
