from unittest.mock import create_autospec

import pytest
import random

from context_engine.chat_engine import ChatEngine
from context_engine.chat_engine.prompt_builder.prompt_builder import BasePromptBuilder
from context_engine.chat_engine.query_generator import QueryGenerator
from context_engine.context_engine import ContextEngine
from context_engine.context_engine.models import ContextQueryResult, ContextSnippet
from context_engine.llm import BaseLLM
from context_engine.llm.models import UserMessage, SystemMessage
from context_engine.models.data_models import MessageBase, Role, Query, Context
from ..stubs.stub_tokenizer import StubTokenizer
from .. import random_words

MOCK_SYSTEM_PROMPT = "This is my mock prompt"
MAX_PROMPT_TOKENS = 100


class TestChatEngine:

    # @classmethod
    # def setup_class(cls):
    #     # cls.mock_prompt_builder = create_autospec(BasePromptBuilder)
    #     cls.mock_llm = create_autospec(BaseLLM)
    #     cls.mock_query_builder = create_autospec(QueryGenerator)
    #     cls.mock_context_engine = create_autospec(ContextEngine)

    def setup(self):
        # self.mock_prompt_builder = create_autospec(BasePromptBuilder)
        self.mock_llm = create_autospec(BaseLLM)
        self.mock_query_builder = create_autospec(QueryGenerator)
        self.mock_context_engine = create_autospec(ContextEngine)

    def _init_chat_engine(self,
                          system_prompt: str = MOCK_SYSTEM_PROMPT,
                          max_prompt_tokens: int = MAX_PROMPT_TOKENS,
                          max_context_tokens: int = None,
                          max_generated_tokens: int = 200,
                          **kwargs):
        return ChatEngine(
            llm=self.mock_llm,
            context_engine=self.mock_context_engine,
            query_builder=self.mock_query_builder,
            tokenizer=StubTokenizer(),
            system_prompt=system_prompt,
            max_prompt_tokens=max_prompt_tokens,
            max_context_tokens=max_context_tokens,
            max_generated_tokens=max_generated_tokens,
            **kwargs
        )

    @staticmethod
    def _generate_text(num_words: int):
        return " ".join(random.choices(random_words, k=num_words))

    def test_chat_default_params(self):
        chat_engine = self._init_chat_engine()

        # Mock input and expected output
        messages = [UserMessage(content="How does photosynthesis work?")]
        mock_queries = [Query(text="How does photosynthesis work?")]
        mock_context = Context(
            content=ContextQueryResult(
                query="How does photosynthesis work?",
                snippets=[ContextSnippet(reference="ref 1",
                                         text=self._generate_text(10)),
                          ContextSnippet(reference="ref 2",
                                         text=self._generate_text(12))]
            ),
            num_tokens=22 + 6
        )
        expected_prompt = [SystemMessage(
            content=MOCK_SYSTEM_PROMPT + f"\nContext: {mock_context.to_text()}"
        )] + messages
        mock_chat_response = "Photosynthesis is a process used by plants..."

        # Set the return values of the mocked methods
        self.mock_query_builder.generate.return_value = mock_queries
        self.mock_context_engine.query.return_value = mock_context
        self.mock_llm.chat_completion.return_value = mock_chat_response

        # Call the method under test
        response = chat_engine.chat(messages)

        # Assertions
        self.mock_query_builder.generate.assert_called_once_with(
            messages,
            max_prompt_tokens=MAX_PROMPT_TOKENS
        )
        self.mock_context_engine.query.assert_called_once_with(
            mock_queries,
            max_context_tokens=70
        )
        self.mock_llm.chat_completion.assert_called_once_with(
            expected_prompt,
            max_tokens=200,
            stream=False,
            model_params=None
        )

        assert response == mock_chat_response

    # @staticmethod
    # @pytest.mark.parametrize("input_len, expected, chat_engine", [
    #     # History length of StubTokenizer is num_words + 3 (hardcoded)
    #     [10, MAX_PROMPT_TOKENS - (10 + 3), {'max_context_tokens': None}],
    #     [30, 80, {'max_context_tokens': None}],
    #     [15, MAX_PROMPT_TOKENS - (15 + 3), {'max_context_tokens': 40}],
    #     [80, 40, {'max_context_tokens': 40}],
    # ], ids=[
    #     "short_history",
    #     "long_history",
    #     "short_history_low_context_budget",
    #     "long_history_low_context_budget"
    # ], indirect=["chat_engine"])
    # def test__calculate_max_context_tokens(input_len,
    #                                        expected,
    #                                        chat_engine
    #                                        ):
    #
    #     messages = [
    #         UserMessage(content=" ".join(["word"] * input_len))
    #     ]
    #     context_len = chat_engine._calculate_max_context_tokens(messages)
    #
    #     #  StubTokenizer adds 3 tokens to the system prompt (treating it as a message)
    #     expected_len = expected - (len(MOCK_SYSTEM_PROMPT.split()) + 3)
    #     assert context_len == expected_len
    #
    # @staticmethod
    # def test__calculate_max_context_tokens_raises(chat_engine):
    #     chat_engine.system_prompt_template = " ".join(["word"] * 91)
    #     messages = [
    #         UserMessage(content=" ".join(["word"] * 10))
    #     ]
    #     with pytest.raises(ValueError):
    #         chat_engine._calculate_max_context_tokens(messages)

    # @staticmethod
    # def test_get_context(chat_engine,
    #                      mock_query_builder,
    #                      mock_context_engine,
    #                      mock_llm
    #                      ):
    #     # TODO: remove code duplication
    #     messages = [MessageBase(role=Role.USER,
    #                             content="How does photosynthesis work?")]
    #     mock_queries = [Query(text="How does photosynthesis work?")]
    #     mock_context = Context(
    #         content=ContextQueryResult(
    #             query="How does photosynthesis work?",
    #             snippets=[ContextSnippet(reference="ref 1", text="cat cat"),
    #                       ContextSnippet(reference="ref 2", text="dog dog")]
    #         ),
    #         num_tokens=20
    #     )
    #
    #     mock_query_builder.generate.return_value = mock_queries
    #     mock_context_engine.query.return_value = mock_context
    #
    #     context = chat_engine.get_context(messages)
    #     mock_query_builder.generate.assert_called_once_with(
    #         messages,
    #         max_prompt_tokens=MAX_PROMPT_TOKENS
    #     )
    #     mock_context_engine.query.assert_called_once_with(
    #         mock_queries,
    #         max_context_tokens=MAX_PROMPT_TOKENS
    #     )
    #
    #     assert context == mock_context

    @pytest.mark.asyncio
    async def test_aget_context_raise(self):
        chat_engine = self._init_chat_engine()
        with pytest.raises(NotImplementedError):
            await chat_engine.aget_context([])

    @pytest.mark.asyncio
    async def test_achat_raise(self):
        chat_engine = self._init_chat_engine()
        with pytest.raises(NotImplementedError):
            await chat_engine.achat([])
