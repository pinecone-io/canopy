from abc import ABC, abstractmethod
from typing import List

from context_engine.chat_engine.exceptions import InvalidRequestError
from context_engine.chat_engine.history_builder.base import BaseHistoryBuilder
from context_engine.context_engine.context_builder.base import BaseContextBuilder
from context_engine.knoweldge_base.models import QueryResult
from context_engine.knoweldge_base.tokenizer.base import Tokenizer
from context_engine.models.data_models import Messages, Role, MessageBase


class BasePromptBuilder(ABC):

    def __init__(self,
                 system_message: str,
                 context_builder: BaseContextBuilder,
                 history_builder: BaseHistoryBuilder,
                 tokenizer: Tokenizer):
        self._system_message = system_message
        self.context_builder = context_builder
        self.history_builder = history_builder
        self._tokenizer = tokenizer

    @abstractmethod
    def build(self,
              messages: Messages,
              query_results: List[QueryResult],
              max_tokens: int) -> Messages:
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

    def __init__(self,
                 system_message: str,
                 context_builder: BaseContextBuilder,
                 history_builder: BaseHistoryBuilder,
                 tokenizer: Tokenizer,
                 context_ratio: float = 0.5):
        super().__init__(system_message,
                         context_builder,
                         history_builder,
                         tokenizer)
        self._context_ratio = context_ratio

    def build(self,
              messages: Messages,
              query_results: List[QueryResult],
              max_tokens: int) -> Messages:
        prompt_massages = [MessageBase(role=Role.SYSTEM,
                                       content=self._system_message)]
        prompt_tokens = self._count_tokens(prompt_massages)
        if prompt_tokens > max_tokens:
            raise InvalidRequestError(
                f'System message tokens {prompt_tokens} exceed max tokens {max_tokens}'
            )

        max_history_tokens = int((max_tokens - prompt_tokens)
                                 * (1.0 - self._context_ratio))
        history, num_tokens = self.history_builder.build(messages,
                                                         max_history_tokens)
        prompt_massages.extend(history)
        context_tokens = max_tokens - self._count_tokens(prompt_massages)
        context = self.context_builder.build(query_results, context_tokens)
        prompt_massages.append(MessageBase(role=Role.SYSTEM,
                                           content=context.to_text()))
        return prompt_massages

    async def abuild(self,
                     messages: Messages,
                     max_tokens: int) -> Messages:
        raise NotImplementedError()
