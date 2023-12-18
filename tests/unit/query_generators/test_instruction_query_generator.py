from typing import List
from unittest.mock import create_autospec

import pytest

from canopy.chat_engine.query_generator import InstructionQueryGenerator
from canopy.llm import BaseLLM
from canopy.models.api_models import ChatResponse, _Choice, TokenCounts
from canopy.models.data_models import Query, UserMessage, AssistantMessage


@pytest.fixture
def mock_llm():
    return create_autospec(BaseLLM)


@pytest.fixture
def query_generator(mock_llm):
    query_gen = InstructionQueryGenerator(
        llm=mock_llm,
    )
    return query_gen


@pytest.fixture
def sample_messages():
    return [UserMessage(content="How can I init a client?"),
            AssistantMessage(content="Which kind of client?"),
            UserMessage(content="A pinecone client.")]


@pytest.mark.parametrize(("response", "query", "call_count"), [
    (
            '{"question": "How do I init a pinecone client?"}',
            "How do I init a pinecone client?",
            1
    ),

    (
            'Unparseable JSON response from LLM, falling back to the last message',
            "A pinecone client.",
            3
    )

])
def test_generate(query_generator,
                  mock_llm,
                  sample_messages,
                  response,
                  query,
                  call_count):
    mock_llm.chat_completion.return_value = ChatResponse(
        id="meta-llama/Llama-2-7b-chat-hf-HTQ-4",
        object="text_completion",
        created=1702569324,
        model='meta-llama/Llama-2-7b-chat-hf',
        usage=TokenCounts(
            prompt_tokens=367,
            completion_tokens=19,
            total_tokens=386
        ),
        choices=[
            _Choice(
                index=0,
                message=AssistantMessage(
                    content=response
                )
            )
        ]
    )

    result = query_generator.generate(messages=sample_messages,
                                      max_prompt_tokens=4096)

    assert mock_llm.chat_completion.call_count == call_count
    assert isinstance(result, List)
    assert len(result) == 1
    assert result[0] == Query(text=query)


@pytest.mark.asyncio
async def test_agenerate_not_implemented(query_generator,
                                         mock_llm,
                                         sample_messages
                                         ):
    with pytest.raises(NotImplementedError):
        await query_generator.agenerate(messages=sample_messages,
                                        max_prompt_tokens=100)
