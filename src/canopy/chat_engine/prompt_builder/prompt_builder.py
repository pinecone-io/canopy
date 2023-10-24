from abc import ABC, abstractmethod

from canopy.chat_engine.exceptions import InvalidRequestError
from canopy.chat_engine.history_pruner import (RaisingHistoryPruner,
                                               RecentHistoryPruner, )
from canopy.chat_engine.history_pruner.base import HistoryPruner
from canopy.chat_engine.models import HistoryPruningMethod
from canopy.tokenizer import Tokenizer
from canopy.models.data_models import Messages, Role, MessageBase


class BasePromptBuilder(ABC):

    def __init__(self,
                 history_pruning: HistoryPruningMethod,
                 min_history_messages: int
                 ):
        self._tokenizer = Tokenizer()
        self._history_pruner: HistoryPruner
        if history_pruning == HistoryPruningMethod.RAISE:
            self._history_pruner = RaisingHistoryPruner(min_history_messages)
        elif history_pruning == HistoryPruningMethod.RECENT:
            self._history_pruner = RecentHistoryPruner(min_history_messages)
        else:
            raise ValueError(f"Unknown history pruning method "
                             f"{history_pruning}.")

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
