import pytest

from canopy.chat_engine.history_pruner import RaisingHistoryPruner
from canopy.models.data_models import UserMessage, AssistantMessage
from canopy.tokenizer import Tokenizer


@pytest.fixture
def raising_history_pruner():
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


def test_build_within_limit(raising_history_pruner, sample_messages):
    token_limit = 50
    messages = raising_history_pruner.build(sample_messages, token_limit)
    assert messages == sample_messages
    assert Tokenizer().messages_token_count(messages) <= token_limit


@pytest.mark.parametrize(
    "token_limit",
    [10, 18],
    ids=["low_limit", "moderate_limit"]
)
def test_build_exceeds_limit(raising_history_pruner, sample_messages, token_limit):
    with pytest.raises(ValueError) as e:
        raising_history_pruner.build(sample_messages, token_limit)
        err_msg = e.value.args[0]
        assert f"history require more than {token_limit} tokens" in err_msg


@pytest.mark.asyncio
async def test_abuild_not_implemented(raising_history_pruner, sample_messages):
    with pytest.raises(NotImplementedError):
        await raising_history_pruner.abuild(sample_messages, 25)
