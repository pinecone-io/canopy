import pytest

from canopy.models.data_models import UserMessage, AssistantMessage


@pytest.fixture
def messages():
    # Create a list of MessageBase objects
    return [
        UserMessage(content="Hello, assistant."),
        AssistantMessage(content="Hello, user. How can I assist you?"),
        UserMessage(content="Just checking in. Be concise."),
    ]
