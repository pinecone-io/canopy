import pytest

from canopy.chat_engine.query_generator.cohere import CohereQueryGenerator
from canopy.models.data_models import MessageBase, Role


@pytest.fixture
def messages():
    return [
        MessageBase(
            role=Role.USER, content="Hello, assistant."),
        MessageBase(
            role=Role.ASSISTANT, content="Hello, user. How can I assist you?"),
        MessageBase(
            role=Role.USER, content="How do I init a pinecone client?.")
    ]


def test_generate_queries(messages):
    query_generator = CohereQueryGenerator()
    queries = query_generator.generate(messages, max_prompt_tokens=100)
    assert queries
    assert queries[0].text


def test_max_tokens_exceeded_raises_error(messages):
    query_generator = CohereQueryGenerator()

    with pytest.raises(ValueError):
        query_generator.generate(messages, max_prompt_tokens=10)
