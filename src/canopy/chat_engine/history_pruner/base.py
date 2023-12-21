from abc import ABC, abstractmethod
from typing import Optional

from canopy.tokenizer import Tokenizer
from canopy.models.data_models import Messages, SystemMessage, Context
from canopy.utils.config import ConfigurableMixin


class HistoryPruner(ABC, ConfigurableMixin):

    def __init__(self):
        self._tokenizer = Tokenizer()

    @abstractmethod
    def build(self,
              chat_history: Messages,
              max_tokens: int,
              system_prompt: Optional[str] = None,
              context: Optional[Context] = None,
              ) -> Messages:
        raise NotImplementedError

    async def abuild(self,
                     chat_history: Messages,
                     max_tokens: int) -> Messages:
        raise NotImplementedError()

    def _max_tokens_history(self,
                            max_tokens: int,
                            system_prompt: Optional[str] = None,
                            context: Optional[Context] = None, ) -> int:
        if system_prompt is not None:
            max_tokens -= self._tokenizer.messages_token_count(
                [SystemMessage(content=system_prompt)]
            )

        if context is not None:
            max_tokens -= context.num_tokens

        return max_tokens
