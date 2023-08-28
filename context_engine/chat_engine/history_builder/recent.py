from typing import Tuple

from context_engine.chat_engine.history_builder.base import BaseHistoryBuilder
from context_engine.models.data_models import Messages, MessageBase


class RecentHistoryBuilder(BaseHistoryBuilder):

    def build(self,
              full_history: Messages,
              max_tokens: int) -> Tuple[Messages, int]:
        truncated_history: Messages = []

        # Start from the most recent message
        for message in reversed(full_history):
            token_count = self._tokenizer.messages_token_count(
                truncated_history + [message]
            )

            # If the message can fit into the remaining tokens, add it
            if token_count <= max_tokens:
                truncated_history.append(message)

            # If the message can't fit, truncate the message
            else:
                empty_message = MessageBase(role=message.role,
                                            content="")
                token_count = self._tokenizer.messages_token_count(
                    truncated_history + [empty_message]
                )
                if token_count > max_tokens:
                    break

                remaining_tokens = max_tokens - token_count
                tokens = self._tokenizer.tokenize(message.content)[:remaining_tokens]
                truncated_content = self._tokenizer.detokenize(tokens)
                truncated_history.append(MessageBase(role=message.role,
                                                     content=truncated_content))
                break
        token_count = self._tokenizer.messages_token_count(truncated_history)
        # Since we started with the most recent messages, we reverse again
        return list(reversed(truncated_history)), token_count

    async def abuild(self,
                     full_history: Messages,
                     max_tokens: int) -> Tuple[Messages, int]:
        raise NotImplementedError
