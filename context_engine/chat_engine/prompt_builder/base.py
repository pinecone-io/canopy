from abc import ABC, abstractmethod
from typing import List, Optional

from context_engine.chat_engine.exceptions import InvalidRequestError
from context_engine.chat_engine.history_builder.base import BaseHistoryBuilder
from context_engine.context_engine.context_builder.base import BaseContextBuilder
from context_engine.knoweldge_base.models import QueryResult
from context_engine.knoweldge_base.tokenizer.base import Tokenizer
from context_engine.models.data_models import Messages, Role, MessageBase


class BasePromptBuilder(ABC):

    def __init__(self,
                 context_builder: BaseContextBuilder,
                 history_builder: BaseHistoryBuilder,
                 tokenizer: Tokenizer):
        self.context_builder = context_builder
        self.history_builder = history_builder
        self._tokenizer = tokenizer

    @abstractmethod
    def build(self,
              system_message: str,
              history: Messages,
              query_results: Optional[List[QueryResult]],
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
                 context_builder: BaseContextBuilder,
                 history_builder: BaseHistoryBuilder,
                 tokenizer: Tokenizer,
                 context_ratio: float):
        super().__init__(context_builder,
                         history_builder,
                         tokenizer)
        self._context_ratio = context_ratio

    def build(self,
              system_message: str,
              history: Messages,
              query_results: Optional[List[QueryResult]],
              max_tokens: int) -> Messages:
        prompt_massages = [MessageBase(role=Role.SYSTEM,
                                       content=system_message)]
        system_tokens = self._count_tokens(prompt_massages)
        if system_tokens > max_tokens:
            raise InvalidRequestError(
                f'System message tokens {system_tokens} exceed max tokens {max_tokens}'
            )

        if query_results is None or len(query_results) == 0:
            max_history_tokens = max_tokens - system_tokens
        else:
            max_history_tokens = int((max_tokens - system_tokens)
                                     * (1.0 - self._context_ratio))

        history, num_tokens = self.history_builder.build(history,
                                                         max_history_tokens)
        prompt_massages.extend(history)

        if query_results is not None and len(query_results) > 0:
            context_tokens = max_tokens - self._count_tokens(prompt_massages)
            context = self.context_builder.build(query_results, context_tokens)
            prompt_massages.append(MessageBase(role=Role.SYSTEM,
                                               content=context.to_text()))
        return prompt_massages

    async def abuild(self,
                     messages: Messages,
                     max_tokens: int) -> Messages:
        raise NotImplementedError()
