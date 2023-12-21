import pytest

from canopy.chat_engine.history_pruner import RecentHistoryPruner
from canopy.models.data_models import UserMessage, \
    AssistantMessage, Context, StringContextContent
from canopy.tokenizer import Tokenizer


SAMPLE_CONTEXT = Context(content=StringContextContent(
    __root__="Some context information"), num_tokens=3
)
SYSTEM_PROMPT = "This is a system prompt."


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
    "token_limit, expected_tail, expected_token_count, context, prompt",
    [
        (50, 5, 33, None, None),
        (18, 2, 11, None, None),
        (10, 1, 6, None, None),
        (50, 5, 33, SAMPLE_CONTEXT, None),
        (50, 5, 33, None, SYSTEM_PROMPT),
        (50, 5, 33, SAMPLE_CONTEXT, SYSTEM_PROMPT),
        (11, 1, 6, SAMPLE_CONTEXT, None),
        (18, 1, 6, None, SYSTEM_PROMPT),
        (19, 1, 6, SAMPLE_CONTEXT, SYSTEM_PROMPT),
    ],
    ids=[
        "full_history_fit_no_context_no_prompt",
        "truncated_no_context_no_prompt",
        "single_message_no_context_no_prompt",
        "full_history_fit_with_context",
        "full_history_fit_with_prompt",
        "full_history_fit_with_context_and_prompt",
        "truncated_with_context",
        "truncated_with_prompt",
        "truncated_with_context_and_prompt",
    ]
)
def test_build(recent_history_builder,
               sample_messages,
               token_limit,
               expected_tail,
               expected_token_count,
               context,
               prompt):
    messages = recent_history_builder.build(sample_messages,
                                            token_limit,
                                            system_prompt=prompt,
                                            context=context)
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
