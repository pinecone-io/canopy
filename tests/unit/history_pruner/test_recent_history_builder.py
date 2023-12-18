import pytest

from canopy.chat_engine.history_pruner import RecentHistoryPruner
from canopy.models.data_models import UserMessage, AssistantMessage
from canopy.tokenizer import Tokenizer


@pytest.fixture
def recent_history_builder():
    return RecentHistoryPruner(min_history_messages=1)


@pytest.fixture
def sample_messages():
    return [
        UserMessage(content="Hello there!"),
        AssistantMessage(content="Hi! How can I help you?"),
        UserMessage(content="Tell me about the weather."),
        AssistantMessage(content="Anything else?"),
        UserMessage(content="No that's enough"),
    ]


@pytest.mark.parametrize(
    "token_limit, expected_tail, expected_token_count",
    [
        (50, 5, 33),  # All messages fit
        (18, 2, 11),  # Only last 2
        (10, 1, 6),  # Only last one

    ],
    ids=[
        "full_history_fit",
        "truncated",
        "multiple_message_truncation",
    ]
)
def test_build(recent_history_builder,
               sample_messages,
               token_limit,
               expected_tail,
               expected_token_count
               ):
    messages = recent_history_builder.build(sample_messages, token_limit)
    assert messages == sample_messages[-expected_tail:]
    assert Tokenizer().messages_token_count(messages) == expected_token_count


def test_min_history_messages(sample_messages):
    recent_history_builder = RecentHistoryPruner(
        min_history_messages=2
    )
    token_limit = 18
    messages = recent_history_builder.build(sample_messages, token_limit)
    assert messages == sample_messages[-2:]
    assert Tokenizer().messages_token_count(messages) == 11

    token_limit = 10
    with pytest.raises(ValueError) as e:
        recent_history_builder.build(sample_messages, token_limit)
        err_msg = e.value.args[0]
        assert f"The {2} most recent" in err_msg
        assert f"calculated history of {token_limit}" in err_msg
        assert "history require 11 tokens" in err_msg


def test_build_with_empty_history(recent_history_builder):
    messages = recent_history_builder.build([], 15)
    assert messages == []
    assert Tokenizer().messages_token_count(messages) == 0


@pytest.mark.asyncio
async def test_abuild_not_implemented(recent_history_builder, sample_messages):
    with pytest.raises(NotImplementedError):
        await recent_history_builder.abuild(sample_messages, 25)
