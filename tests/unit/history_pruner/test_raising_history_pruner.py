import pytest

from canopy.chat_engine.history_pruner import RaisingHistoryPruner
from canopy.models.data_models import \
    UserMessage, AssistantMessage, Context, StringContextContent
from canopy.tokenizer import Tokenizer


SAMPLE_CONTEXT = Context(content=StringContextContent(
    __root__="Some context information"), num_tokens=3
)
SYSTEM_PROMPT = "This is a system prompt."


@pytest.fixture
def raising_history_builder():
    return RaisingHistoryPruner()


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
    "token_limit, expected_token_count, context, prompt",
    [
        (33, 33, None, None),
        (50, 33, SAMPLE_CONTEXT, None),
        (50, 33, None, SYSTEM_PROMPT),
        (50, 33, SAMPLE_CONTEXT, SYSTEM_PROMPT),
    ],
    ids=[
        "within_limit_no_context_no_prompt",
        "within_limit_with_context",
        "within_limit_with_prompt",
        "within_limit_with_context_and_prompt",
    ]
)
def test_build_within_limits(raising_history_builder, sample_messages,
                             token_limit, expected_token_count, context, prompt):
    messages = raising_history_builder.build(sample_messages, token_limit,
                                             system_prompt=prompt, context=context)
    assert Tokenizer().messages_token_count(messages) == expected_token_count


@pytest.mark.parametrize(
    "token_limit, context, prompt",
    [
        (32, None, None),
        (33, SAMPLE_CONTEXT, None),
        (33, None, SYSTEM_PROMPT),
        (31, SAMPLE_CONTEXT, SYSTEM_PROMPT),
    ],
    ids=[
        "exceed_limit_no_context_no_prompt",
        "exceed_limit_with_context",
        "exceed_limit_with_prompt",
        "exceed_limit_with_context_and_prompt",
    ]
)
def test_build_exceeds_limits(raising_history_builder, sample_messages,
                              token_limit, context, prompt):
    with pytest.raises(ValueError) as e:
        raising_history_builder.build(sample_messages, token_limit,
                                      system_prompt=prompt, context=context)
        err_msg = e.value.args[0]
        assert f"require {Tokenizer().messages_token_count(sample_messages)} " \
               f"tokens" in err_msg
        assert f"of {token_limit} tokens left for history" in err_msg


@pytest.mark.asyncio
async def test_abuild_not_implemented(raising_history_builder, sample_messages):
    with pytest.raises(NotImplementedError):
        await raising_history_builder.abuild(sample_messages, 25)
