from typing import Tuple, Optional

from canopy.chat_engine.history_pruner.base import HistoryPruner
from canopy.models.data_models import Messages


class RecentHistoryPruner(HistoryPruner):

    def __init__(self,
                 min_history_messages: int = 1):
        super().__init__()
        self._min_history_messages = min_history_messages

    def build(self,
              chat_history: Messages,
              max_tokens: int,
              system_prompt: Optional[str] = None,
              context: Optional[str] = None,
              ) -> Tuple[Messages, int]:
        max_tokens_history = self._max_tokens_history(max_tokens, system_prompt, context)
        token_count = self._tokenizer.messages_token_count(chat_history)
        if token_count < max_tokens:
            return chat_history, token_count

        truncated_history = chat_history[-self._min_history_messages:]
        token_count = self._tokenizer.messages_token_count(truncated_history)
        if token_count > max_tokens:
            raise ValueError(f"The {self._min_history_messages} most recent messages in"
                             f" history require {token_count} tokens, which exceeds the"
                             f" calculated limit for history of {max_tokens_history} tokens out of total {max_tokens} allowed in context window.")

        for message in reversed(chat_history[:-self._min_history_messages]):
            token_count = self._tokenizer.messages_token_count(
                truncated_history + [message]
            )

            # If the message can fit into the remaining tokens, add it
            if token_count > max_tokens:
                break

            truncated_history.insert(0, message)

        token_count = self._tokenizer.messages_token_count(truncated_history)

        return truncated_history, token_count

    async def abuild(self,
                     chat_history: Messages,
                     max_tokens: int) -> Tuple[Messages, int]:
        raise NotImplementedError()
