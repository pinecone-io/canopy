from unittest.mock import MagicMock

import pytest
from cohere.error import CohereAPIError

from canopy.models.data_models import Context, ContextContent, Role, MessageBase
from canopy.context_engine.context_builder.stuffing import (
    StuffingContextContent, ContextQueryResult, ContextSnippet
)
from canopy.models.api_models import ChatResponse, StreamingChatChunk
from canopy.llm.cohere import CohereLLM


def assert_chat_completion(response):
    assert len(response.choices) == 1  # Cohere API does not return multiple choices.

    assert isinstance(response.choices[0].message, MessageBase)
    assert isinstance(response.choices[0].message.content, str)
    assert len(response.choices[0].message.content) > 0
    assert isinstance(response.choices[0].message.role, Role)


@pytest.fixture
def model_name():
    return "command"


@pytest.fixture
def messages():
    return [
        MessageBase(
            role=Role.USER, content="Hello, assistant."),
        MessageBase(
            role=Role.ASSISTANT, content="Hello, user. How can I assist you?"),
        MessageBase(
            role=Role.USER, content="Tell me about X.")
    ]


@pytest.fixture
def system_prompt():
    return "Use only the provided documents to answer."


@pytest.fixture
def expected_chat_kwargs():
    return {
        "model": "command",
        "message": "Tell me about X.",
        "chat_history": [
            {'role': 'USER', 'message': 'Hello, assistant.'},
            {"role": "CHATBOT", "message": "Hello, user. How can I assist you?"}
        ],
        "connectors": None,
        "documents": [],
        "preamble_override": "Use only the provided documents to answer.",
        "stream": False,
        "max_tokens": None,
    }


@pytest.fixture
def model_params_high_temperature():
    return {"temperature": 0.9, "p": 0.95}


@pytest.fixture
def model_params_low_temperature():
    return {"temperature": 0.2, "p": 0.5}


@pytest.fixture
def cohere_llm(model_name):
    return CohereLLM(model_name=model_name)


@pytest.fixture
def unsupported_context():
    class UnsupportedContextContent(ContextContent):
        def to_text(self, **kwargs):
            return ''

    return Context(content=UnsupportedContextContent(), num_tokens=123)


def test_init_with_custom_params():
    llm = CohereLLM(model_name="test_model_name",
                    api_key="test_api_key",
                    temperature=0.9)

    assert llm.model_name == "test_model_name"
    assert llm.default_model_params["temperature"] == 0.9
    assert llm._client.api_key == "test_api_key"


def test_chat_completion(cohere_llm, messages, system_prompt, expected_chat_kwargs):
    cohere_llm._client = MagicMock(wraps=cohere_llm._client)
    response = cohere_llm.chat_completion(
        chat_history=messages, system_prompt=system_prompt)
    cohere_llm._client.chat.assert_called_once_with(**expected_chat_kwargs)
    assert_chat_completion(response)


def test_chat_completion_high_temperature(cohere_llm,
                                          messages,
                                          model_params_high_temperature):
    response = cohere_llm.chat_completion(
        chat_history=messages,
        model_params=model_params_high_temperature,
        system_prompt='',
    )
    assert_chat_completion(response)


def test_chat_completion_low_temperature(cohere_llm,
                                         messages,
                                         model_params_low_temperature):
    response = cohere_llm.chat_completion(chat_history=messages,
                                          model_params=model_params_low_temperature,
                                          system_prompt='')
    assert_chat_completion(response)


def test_chat_streaming(cohere_llm, messages):
    stream = True
    response = cohere_llm.chat_completion(chat_history=messages,
                                          stream=stream,
                                          system_prompt='')
    messages_received = [message for message in response]
    assert len(messages_received) > 0

    for message in messages_received:
        assert isinstance(message, StreamingChatChunk)
        assert message.object == "chat.completion.chunk"


def test_max_tokens(cohere_llm, messages):
    max_tokens = 2
    response = cohere_llm.chat_completion(chat_history=messages,
                                          max_tokens=max_tokens,
                                          system_prompt='')
    assert isinstance(response, ChatResponse)
    assert len(response.choices[0].message.content.split()) <= max_tokens


def test_missing_messages(cohere_llm):
    with pytest.raises(CohereAPIError):
        cohere_llm.chat_completion(chat_history=[], system_prompt='')


def test_negative_max_tokens(cohere_llm, messages):
    with pytest.raises(CohereAPIError):
        cohere_llm.chat_completion(
            chat_history=messages, max_tokens=-5, system_prompt='')


def test_chat_completion_api_failure_propagates(cohere_llm,
                                                messages):
    cohere_llm._client = MagicMock()
    cohere_llm._client.chat.side_effect = CohereAPIError("API call failed")

    with pytest.raises(CohereAPIError, match="API call failed"):
        cohere_llm.chat_completion(chat_history=messages, system_prompt="")


def test_chat_completion_with_unsupported_context_engine(cohere_llm,
                                                         messages,
                                                         unsupported_context):
    cohere_llm._client = MagicMock()

    with pytest.raises(NotImplementedError):
        cohere_llm.chat_completion(chat_history=messages,
                                   system_prompt="",
                                   context=unsupported_context)


def test_chat_completion_with_stuffing_context_snippets(cohere_llm,
                                                        messages,
                                                        expected_chat_kwargs,
                                                        system_prompt):
    cohere_llm._client = MagicMock(wraps=cohere_llm._client)
    content = StuffingContextContent(__root__=[
        ContextQueryResult(query="", snippets=[
            ContextSnippet(
                source="https://www.example.com/document",
                text="Document text",
            ),
            ContextSnippet(
                source="https://www.example.com/second_document",
                text="Second document text",
            )
        ])
    ])
    stuffing_context = Context(
        content=content,
        num_tokens=123)

    response = cohere_llm.chat_completion(
        chat_history=messages,
        system_prompt=system_prompt,
        context=stuffing_context)

    # Check that we got a valid chat response - details tested in other tests
    assert isinstance(response, ChatResponse)
    assert response.object == "chat.completion"

    # Check that Cohere client was called with the snippets
    expected_chat_kwargs["documents"] = [
        {
            "source": "https://www.example.com/document",
            "text": "Document text",
        },
        {
            "source": "https://www.example.com/second_document",
            "text": "Second document text",
        },
    ]
    cohere_llm._client.chat.assert_called_once_with(**expected_chat_kwargs)


def test_token_counts_mapped_in_chat_response(cohere_llm, messages, system_prompt):
    response = cohere_llm.chat_completion(chat_history=messages,
                                          system_prompt=system_prompt)
    assert response.usage.prompt_tokens == 45
    assert response.usage.completion_tokens
    assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
    )
