from typing import Optional

from canopy.chat_engine.history_pruner.base import HistoryPruner
from canopy.models.data_models import Messages, Context


class RaisingHistoryPruner(HistoryPruner):

    def build(self,
              chat_history: Messages,
              max_tokens: int,
              system_prompt: Optional[str] = None,
              context: Optional[Context] = None, ) -> Messages:
        max_tokens = self._max_tokens_history(max_tokens,
                                              system_prompt,
                                              context)
        token_count = self._tokenizer.messages_token_count(chat_history)
        if token_count > max_tokens:
            raise ValueError(f"The history require {token_count} tokens, "
                             f"which exceeds the calculated limit for history "
                             f"of {max_tokens} tokens left for"
                             f" history out of {max_tokens} tokens"
                             f" allowed in context window.")
        return chat_history

    async def abuild(self,
                     chat_history: Messages,
                     max_tokens: int) -> Messages:
        raise NotImplementedError()
