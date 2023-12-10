import pytest

from canopy.chat_engine.query_generator import LastMessageQueryGenerator
from canopy.models.data_models import UserMessage, Query


@pytest.fixture
def sample_messages():
    return [
        UserMessage(content="What is photosynthesis?")
    ]


@pytest.fixture
def query_generator():
    return LastMessageQueryGenerator()


def test_generate(query_generator, sample_messages):
    expected = [Query(text=sample_messages[-1].content)]
    actual = query_generator.generate(sample_messages, 0)
    assert actual == expected


@pytest.mark.asyncio
async def test_agenerate(query_generator, sample_messages):
    expected = [Query(text=sample_messages[-1].content)]
    actual = await query_generator.agenerate(sample_messages, 0)
    assert actual == expected
