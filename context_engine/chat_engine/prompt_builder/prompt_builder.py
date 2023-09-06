from abc import ABC, abstractmethod

from context_engine.chat_engine.exceptions import InvalidRequestError
from context_engine.chat_engine.history_builder.base import HistoryBuilder
from context_engine.knoweldge_base.tokenizer.base import Tokenizer
from context_engine.models.data_models import Messages, Role, MessageBase


class BasePromptBuilder(ABC):

    def __init__(self,
                 tokenizer: Tokenizer,
                 history_pruner: HistoryBuilder,
                 ):
        self._tokenizer = tokenizer
        self._history_pruner = history_pruner

    @abstractmethod
    def build(self,
              system_prompt: str,
              history: Messages,
              max_tokens: int
              ) -> Messages:
        pass

    @abstractmethod
    async def abuild(self,
                     messages: Messages,
                     max_tokens: int) -> Messages:
        pass

    def _count_tokens(self, messages: Messages) -> int:
        return sum([len(self._tokenizer.tokenize(message.json()))
                    for message in messages])


class PromptBuilder(BasePromptBuilder):

    def build(self,
              system_prompt: str,
              history: Messages,
              max_tokens: int
              ) -> Messages:
        system_massage = [MessageBase(role=Role.SYSTEM,
                                      content=system_prompt)]
        prompt_tokens = self._tokenizer.messages_token_count(system_massage)
        if prompt_tokens > max_tokens:
            raise InvalidRequestError(
                f'System message tokens {prompt_tokens} exceed max tokens {max_tokens}'
            )

        max_history_tokens = max_tokens - prompt_tokens
        pruned_history, num_tokens = self._history_pruner.build(history,
                                                                max_history_tokens)

        return system_massage + pruned_history

    async def abuild(self,
                     messages: Messages,
                     max_tokens: int
                     ) -> Messages:
        raise NotImplementedError()
