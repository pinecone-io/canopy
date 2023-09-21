from unittest.mock import create_autospec

import pytest

from resin.chat_engine.exceptions import InvalidRequestError
from resin.chat_engine.history_pruner.base import HistoryPruner
from resin.chat_engine.models import HistoryPruningMethod
from resin.chat_engine.prompt_builder import PromptBuilder
from resin.models.data_models import MessageBase, Role


@pytest.fixture
def mock_history_builder():
    return create_autospec(HistoryPruner)


@pytest.fixture
def prompt_builder(mock_history_builder):
    pb = PromptBuilder(HistoryPruningMethod.RECENT, 1)
    pb._history_pruner = mock_history_builder
    return pb


def test_build(prompt_builder, mock_history_builder):
    system_message = "Starting message bla"
    prompt_len = len(system_message.split()) + 3
    history = [MessageBase(role=Role.USER, content="Hello")]
    max_tokens = 25

    mock_history_builder.build.return_value = (
        [MessageBase(role=Role.USER, content="Mocked History")], 5)

    messages = prompt_builder.build(
        system_message, history, max_tokens)

    assert mock_history_builder.build.call_count == 1
    mock_history_builder.build.assert_called_once_with(
        history, max_tokens - prompt_len)

    # Adjust assertions as per the behavior expected
    assert len(messages) == 2
    assert messages[0].content == system_message
    assert messages[1].content == "Mocked History"


def test_build_raises(prompt_builder, mock_history_builder):
    system_message = " ".join(["word"] * 20)
    history = [MessageBase(role=Role.USER, content="Hello")]
    max_tokens = 10

    with pytest.raises(InvalidRequestError):
        prompt_builder.build(system_message, history, max_tokens=max_tokens)
