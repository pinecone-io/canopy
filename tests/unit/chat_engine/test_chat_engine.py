from unittest.mock import create_autospec

import pytest

from context_engine.chat_engine.base import ChatEngine
from context_engine.chat_engine.prompt_builder.base import BasePromptBuilder
from context_engine.chat_engine.query_generator.base import QueryGenerator
from context_engine.context_engine import ContextEngine
from context_engine.context_engine.models import ContextQueryResult, ContextSnippet
from context_engine.llm.base import BaseLLM
from context_engine.llm.models import UserMessage
from context_engine.models.data_models import MessageBase, Role, Query, Context
from stubs.stub_tokenizer import StubTokenizer

MOCK_SYSTEM_PROMPT = "This is my mock prompt {context}"
MAX_PROMPT_TOKENS = 100


class TestChatEngine:
    @staticmethod
    @pytest.fixture
    def mock_llm():
        return create_autospec(BaseLLM)

    @staticmethod
    @pytest.fixture
    def mock_query_builder():
        return create_autospec(QueryGenerator)

    @staticmethod
    @pytest.fixture
    def mock_context_engine():
        return create_autospec(ContextEngine)

    @staticmethod
    @pytest.fixture
    def mock_prompt_builder():
        return create_autospec(BasePromptBuilder)

    @staticmethod
    @pytest.fixture
    def stub_tokenizer():
        return StubTokenizer()

    @staticmethod
    @pytest.fixture
    def chat_engine(mock_llm,
                    mock_query_builder,
                    mock_context_engine,
                    mock_prompt_builder,
                    stub_tokenizer
                    ):
        chat_engine = ChatEngine(
            llm=mock_llm,
            context_engine=mock_context_engine,
            query_builder=mock_query_builder,
            tokenizer=stub_tokenizer,
            system_prompt=MOCK_SYSTEM_PROMPT,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            max_generated_tokens=200
        )
        chat_engine._prompt_builder = mock_prompt_builder
        return chat_engine

    @staticmethod
    def test_chat_default_params(chat_engine,
                                 mocker,
                                 mock_query_builder,
                                 mock_context_engine,
                                 mock_prompt_builder,
                                 mock_llm
                                 ):
        # Mock input and expected output
        messages = [MessageBase(role=Role.USER,
                                content="How does photosynthesis work?")]
        mock_queries = [Query(text="How does photosynthesis work?")]
        mock_context = Context(
            content=ContextQueryResult(
                query="How does photosynthesis work?",
                snippets=[ContextSnippet(reference="ref 1", text="cat cat"),
                          ContextSnippet(reference="ref 2", text="dog dog")]
            ),
            num_tokens=20
        )
        mock_prompt_messages = [MessageBase(role=Role.SYSTEM,
                                            content="Some content...")]
        mock_chat_response = "Photosynthesis is a process used by plants..."

        # Set the return values of the mocked methods
        mock_query_builder.generate.return_value = mock_queries
        mock_context_engine.query.return_value = mock_context
        mock_prompt_builder.build.return_value = mock_prompt_messages
        mock_llm.chat_completion.return_value = mock_chat_response
        calc_max_tokens = mocker.patch.object(chat_engine,
                                              "_calculate_max_context_tokens")
        calc_max_tokens.return_value = 50

        # Call the method under test
        response = chat_engine.chat(messages)

        # Assertions
        mock_query_builder.generate.assert_called_once_with(
            messages,
            max_prompt_tokens=MAX_PROMPT_TOKENS
        )
        mock_context_engine.query.assert_called_once_with(mock_queries,
                                                          max_context_tokens=50)
        mock_prompt_builder.build.assert_called_once_with(
            system_prompt=MOCK_SYSTEM_PROMPT + f"\nContext: {mock_context.to_text()}",
            history=messages,
            max_tokens=MAX_PROMPT_TOKENS
        )
        mock_llm.chat_completion.assert_called_once_with(
            mock_prompt_messages,
            max_tokens=200,
            stream=False,
            model_params=None
        )

        assert response == mock_chat_response

    @staticmethod
    @pytest.mark.parametrize("input_len, context_ratio, expected", [
        # History length of StubTokenizer is num_words + 3 (hardcoded)
        [10, 0.8, MAX_PROMPT_TOKENS - (10 + 3)],
        [30, 0.8, 80],
        [15, 0.4, MAX_PROMPT_TOKENS - (15 + 3)],
        [80, 0.4, 40],
    ], ids=[
        "short_history",
        "long_history",
        "short_history_low_ratio",
        "long_history_low_ratio"
    ])
    def test__calculate_max_context_tokens(chat_engine,
                                           # stub_tokenizer,
                                           input_len,
                                           context_ratio,
                                           expected,
                                           ):
        # TODO: refactor test so we can pass `context_ratio` to constructor
        chat_engine._context_to_history_ratio = context_ratio

        messages = [
            UserMessage(content=" ".join(["word"] * input_len))
        ]
        context_len = chat_engine._calculate_max_context_tokens(messages)

        #  StubTokenizer adds 3 tokens to the system prompt (treating it as a message)
        expected_len = expected - (len(MOCK_SYSTEM_PROMPT.split()) + 3)
        assert context_len == expected_len

    @staticmethod
    def test__calculate_max_context_tokens_raises(chat_engine):
        chat_engine.system_prompt_template = " ".join(["word"] * 91)
        messages = [
            UserMessage(content=" ".join(["word"] * 10))
        ]
        with pytest.raises(ValueError):
            chat_engine._calculate_max_context_tokens(messages)

    @staticmethod
    def test_get_context(chat_engine,
                         mock_query_builder,
                         mock_context_engine,
                         mock_llm
                         ):
        # TODO: remove code duplication
        messages = [MessageBase(role=Role.USER,
                                content="How does photosynthesis work?")]
        mock_queries = [Query(text="How does photosynthesis work?")]
        mock_context = Context(
            content=ContextQueryResult(
                query="How does photosynthesis work?",
                snippets=[ContextSnippet(reference="ref 1", text="cat cat"),
                          ContextSnippet(reference="ref 2", text="dog dog")]
            ),
            num_tokens=20
        )

        mock_query_builder.generate.return_value = mock_queries
        mock_context_engine.query.return_value = mock_context

        context = chat_engine.get_context(messages)
        mock_query_builder.generate.assert_called_once_with(
            messages,
            max_prompt_tokens=MAX_PROMPT_TOKENS
        )
        mock_context_engine.query.assert_called_once_with(
            mock_queries,
            max_context_tokens=MAX_PROMPT_TOKENS
        )

        assert context == mock_context

    @staticmethod
    @pytest.mark.asyncio
    async def test_aget_context_raise(chat_engine):
        with pytest.raises(NotImplementedError):
            await chat_engine.aget_context([])

    @staticmethod
    @pytest.mark.asyncio
    async def test_achat_raise(chat_engine):
        with pytest.raises(NotImplementedError):
            await chat_engine.achat([])
