import pytest
from context_engine.chat_engine.history_builder.recent import RecentHistoryBuilder
from context_engine.models.data_models import MessageBase, Role
from tests.unit.stubs.stub_tokenizer import StubTokenizer


class TestRecentHistoryBuilder:

    @staticmethod
    @pytest.fixture
    def recent_history_builder():
        return RecentHistoryBuilder(StubTokenizer())

    @staticmethod
    @pytest.fixture
    def sample_messages():
        return [
            MessageBase(role=Role.USER, content="Hello there!"),
            MessageBase(role=Role.ASSISTANT, content="Hi! How can I help you?"),
            MessageBase(role=Role.USER, content="Tell me about the weather.")
        ]

    @staticmethod
    def test_build_with_full_history_fit(recent_history_builder, sample_messages):
        messages, token_count = recent_history_builder.build(sample_messages, 25)

        assert messages == sample_messages

        # 13 tokens for content + 9 for overhead
        assert token_count == 22

    @staticmethod
    def test_build_with_truncated_full_message(recent_history_builder, sample_messages):
        messages, token_count = recent_history_builder.build(sample_messages, 18)

        expected_messages = [
            MessageBase(role=Role.ASSISTANT, content="Hi! How can I help you?"),
            MessageBase(role=Role.USER, content="Tell me about the weather.")
        ]

        assert messages == expected_messages

        # 11 tokens for content + 6 for overhead
        assert token_count == 17

    @staticmethod
    def test_multiple_message_truncation(recent_history_builder, sample_messages):
        messages, token_count = recent_history_builder.build(sample_messages, 10)

        expected_messages = [
            MessageBase(role=Role.USER, content="Tell me about the weather.")
        ]

        assert messages == expected_messages

        # 5 tokens for content + 3 for overhead
        assert token_count == 8

    @staticmethod
    def test_build_with_truncated_message(recent_history_builder, sample_messages):
        messages, token_count = recent_history_builder.build(sample_messages, 10)

        expected_messages = [
            MessageBase(role=Role.USER, content="Tell me about the weather.")
        ]

        assert messages == expected_messages

        # 5 tokens for content + 3 for overhead
        assert token_count == 8

    @staticmethod
    def test_build_exact_token_fit(recent_history_builder, sample_messages):
        messages, token_count = recent_history_builder.build(sample_messages, 8)

        expected_messages = [
            MessageBase(role=Role.USER, content="Tell me about the weather.")
        ]

        assert messages == expected_messages

        # 5 tokens for content + 3
        assert token_count == 8

    @staticmethod
    def test_build_with_empty_history(recent_history_builder):
        messages, token_count = recent_history_builder.build([], 15)

        assert messages == []

        # 11 tokens for content + 6 for overhead
        assert token_count == 0

    @staticmethod
    def test_build_with_zero_tokens_fit(recent_history_builder, sample_messages):
        messages, token_count = recent_history_builder.build([], 2)

        assert messages == []

        # 11 tokens for content + 6 for overhead
        assert token_count == 0

    @staticmethod
    @pytest.mark.asyncio
    async def test_abuild_not_implemented(recent_history_builder, sample_messages):
        with pytest.raises(NotImplementedError):
            await recent_history_builder.abuild(sample_messages, 25)
