from unittest.mock import create_autospec

import pytest
from context_engine.chat_engine.exceptions import InvalidRequestError
from context_engine.chat_engine.history_builder.base import BaseHistoryBuilder
from context_engine.chat_engine.prompt_builder.base import PromptBuilder
from context_engine.context_engine.context_builder.base import BaseContextBuilder
from context_engine.knoweldge_base.models import QueryResult, DocumentWithScore
from context_engine.models.data_models import MessageBase, Role, Context
from tests.unit.stubs.stub_tokenizer import StubTokenizer


class TestPromptBuilder:

    @staticmethod
    @pytest.fixture
    def mock_context_builder():
        return create_autospec(BaseContextBuilder)

    @staticmethod
    @pytest.fixture
    def mock_history_builder():
        return create_autospec(BaseHistoryBuilder)

    @staticmethod
    @pytest.fixture
    def stub_tokenizer():
        return StubTokenizer()

    @staticmethod
    @pytest.fixture
    def prompt_builder(mock_context_builder, mock_history_builder, stub_tokenizer):
        context_ratio = 0.6
        return PromptBuilder(mock_context_builder,
                             mock_history_builder,
                             stub_tokenizer,
                             context_ratio)

    @staticmethod
    @pytest.fixture
    def mock_query_results():
        sample_text = "Sample context from a document."
        return [
            QueryResult(
                query="mock query",
                documents=[DocumentWithScore(id="1",
                                             text=sample_text,
                                             metadata={},
                                             score=0.95)]
            )
        ]

    @staticmethod
    def test_build_with_none_query_results(prompt_builder,
                                         mock_history_builder,
                                         mock_context_builder):
        system_message = "Starting message"
        history = [MessageBase(role=Role.USER, content="Hello")]
        max_tokens = 25

        mock_history_builder.build.return_value = (
            [MessageBase(role=Role.USER, content="Mocked History")], 5)

        messages = prompt_builder.build(
            system_message, history, None, max_tokens)

        assert mock_history_builder.build.call_count == 1
        mock_history_builder.build.assert_called_once_with(
            history, max_tokens - 5)

        assert mock_context_builder.build.call_count == 0

        # Adjust assertions as per the behavior expected
        assert len(messages) == 2
        assert messages[0].content == system_message
        assert messages[1].content == "Mocked History"

    @staticmethod
    def test_build_with_empty_query_results(prompt_builder,
                                            mock_history_builder,
                                            mock_context_builder):
        system_message = "Starting message"
        history = [MessageBase(role=Role.USER, content="Hello")]
        max_tokens = 25

        mock_history_builder.build.return_value = (
            [MessageBase(role=Role.USER, content="Mocked History")], 5)

        messages = prompt_builder.build(
            system_message, history, [], max_tokens)

        assert mock_history_builder.build.call_count == 1
        mock_history_builder.build.assert_called_once_with(
            history, max_tokens - 5)

        assert mock_context_builder.build.call_count == 0

        # Adjust assertions as per the behavior expected
        assert len(messages) == 2
        assert messages[0].content == system_message
        assert messages[1].content == "Mocked History"

    @staticmethod
    def test_build_with_query_results_all_fit(prompt_builder,
                                              mock_history_builder,
                                              mock_context_builder,
                                              mock_query_results):
        system_message = "Starting message"
        history = [MessageBase(role=Role.USER, content="Hello")]
        max_tokens = 50

        mock_history_builder.build.return_value = (
            [MessageBase(role=Role.USER, content="Mocked History")], 5)

        context_mock = create_autospec(Context)
        mock_context_builder.build.return_value = context_mock
        context_mock.to_text.return_value = "Mock context text"

        messages = prompt_builder.build(
            system_message, history, mock_query_results, max_tokens)

        assert mock_history_builder.build.call_count == 1
        mock_history_builder.build.assert_called_once_with(
            history, int((max_tokens - 5) * 0.4))

        assert mock_context_builder.build.call_count == 1
        mock_context_builder.build.assert_called_once_with(
            mock_query_results, max_tokens - 5 * 2 - 3)

        # Adjust assertions as per the behavior expected
        assert len(messages) == 3
        assert messages[0].content == system_message
        assert messages[1].content == "Mocked History"
        assert messages[2].content == "Mock context text"

    @staticmethod
    def test_build_system_message_exceeds_max_tokens(prompt_builder):
        system_message = "A very long system message that will exceed the allowed token limit for sure." # noqa
        history = [MessageBase(role=Role.USER, content="Hello")]
        max_tokens = 5

        with pytest.raises(InvalidRequestError):
            prompt_builder.build(system_message, history, None, max_tokens)

    @staticmethod
    def test_build_with_empty_history(prompt_builder,
                                      mock_history_builder,
                                      mock_query_results,
                                      mock_context_builder):
        system_message = "Starting message"
        history = []
        max_tokens = 50

        mock_history_builder.build.return_value = ([], 0)

        context_mock = create_autospec(Context)
        mock_context_builder.build.return_value = context_mock
        context_mock.to_text.return_value = "Mock context text"

        messages = prompt_builder.build(
            system_message, history, mock_query_results, max_tokens)

        assert mock_history_builder.build.call_count == 1
        mock_history_builder.build.assert_called_once_with(
            history, int((max_tokens - 5) * 0.4))

        assert mock_context_builder.build.call_count == 1
        mock_context_builder.build.assert_called_once_with(
            mock_query_results, max_tokens - 5 - 3)

        # Adjust assertions as per the behavior expected
        assert len(messages) == 2
        assert messages[0].content == system_message
        assert messages[1].content == "Mock context text"
