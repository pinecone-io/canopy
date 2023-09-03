from typing import Tuple

from context_engine.chat_engine.history_builder.base import BaseHistoryBuilder
from context_engine.models.data_models import Messages


class RecentHistoryBuilder(BaseHistoryBuilder):

    def build(self,
              history: Messages,
              max_tokens: int) -> Tuple[Messages, int]:
        token_count = self._tokenizer.messages_token_count(history)
        if token_count < max_tokens:
            return history, token_count

        truncated_history = history[-self._min_history_messages:]
        token_count = self._tokenizer.messages_token_count(truncated_history)
        if token_count > max_tokens:
            raise ValueError(f"The last {self._min_history_messages} messages in "
                             f"history require {token_count} tokens, "
                             f"which exceeds the calculated limit for history "
                             f"of {max_tokens} tokens.")

        for message in reversed(history):
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
                     full_history: Messages,
                     max_tokens: int) -> Tuple[Messages, int]:
        raise NotImplementedError
