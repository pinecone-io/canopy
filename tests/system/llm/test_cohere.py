from unittest.mock import MagicMock

import pytest
from cohere.error import CohereAPIError

from canopy.models.data_models import Role, MessageBase # noqa
from canopy.models.api_models import ChatResponse, StreamingChatChunk # noqa
from canopy.llm.cohere import CohereLLM  # noqa


def assert_chat_completion(response):
    assert len(response.choices) == 1  # Cohere API does not return multiple choices.

    for choice in response.choices:
        assert isinstance(choice.message, MessageBase)
        assert isinstance(choice.message.content, str)
        assert len(choice.message.content) > 0
        assert isinstance(choice.message.role, Role)


def assert_function_call_format(result):
    assert isinstance(result, dict)
    assert "queries" in result
    assert isinstance(result["queries"], list)
    assert len(result["queries"]) > 0
    assert isinstance(result["queries"][0], str)
    assert len(result["queries"][0]) > 0


class TestCohereLLM:

    @staticmethod
    @pytest.fixture
    def model_name():
        return "command"

    @staticmethod
    @pytest.fixture
    def messages():
        # Create a list of MessageBase objects
        return [
            MessageBase(role=Role.USER, content="Hello, assistant."),
            MessageBase(role=Role.ASSISTANT,
                        content="Hello, user. How can I assist you?")
        ]

    @staticmethod
    @pytest.fixture
    def model_params_high_temperature():
        return {"temperature": 0.9, "p": 0.95}

    @staticmethod
    @pytest.fixture
    def model_params_low_temperature():
        return {"temperature": 0.2, "p": 0.5}

    @staticmethod
    @pytest.fixture
    def cohere_llm(model_name):
        return CohereLLM(model_name=model_name)

    @staticmethod
    def test_init_with_custom_params():
        llm = CohereLLM(model_name="test_model_name",
                        api_key="test_api_key",
                        temperature=0.9)

        assert llm.model_name == "test_model_name"
        assert llm.default_model_params["temperature"] == 0.9
        assert llm._client.api_key == "test_api_key"

    @staticmethod
    def test_chat_completion(cohere_llm, messages):
        response = cohere_llm.chat_completion(chat_history=messages, system_prompt='')
        assert_chat_completion(response)

    @staticmethod
    def test_chat_completion_high_temperature(cohere_llm,
                                              messages,
                                              model_params_high_temperature):
        response = cohere_llm.chat_completion(
            chat_history=messages,
            model_params=model_params_high_temperature,
            system_prompt='',
        )
        assert_chat_completion(response)

    @staticmethod
    def test_chat_completion_low_temperature(cohere_llm,
                                             messages,
                                             model_params_low_temperature):
        response = cohere_llm.chat_completion(chat_history=messages,
                                              model_params=model_params_low_temperature,
                                              system_prompt='')
        assert_chat_completion(response)

    @staticmethod
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

    @staticmethod
    def test_max_tokens(cohere_llm, messages):
        max_tokens = 2
        response = cohere_llm.chat_completion(chat_history=messages,
                                              max_tokens=max_tokens,
                                              system_prompt='')
        assert isinstance(response, ChatResponse)
        assert len(response.choices[0].message.content.split()) <= max_tokens

    @staticmethod
    def test_missing_messages(cohere_llm):
        with pytest.raises(CohereAPIError):
            cohere_llm.chat_completion(chat_history=[], system_prompt='')

    @staticmethod
    def test_negative_max_tokens(cohere_llm, messages):
        with pytest.raises(CohereAPIError):
            cohere_llm.chat_completion(chat_history=messages, max_tokens=-5, system_prompt='')

    @staticmethod
    def test_chat_completion_api_failure_propagates(cohere_llm,
                                                    messages):
        cohere_llm._client = MagicMock()
        cohere_llm._client.chat.side_effect = CohereAPIError("API call failed")

        with pytest.raises(CohereAPIError, match="API call failed"):
            cohere_llm.chat_completion(chat_history=messages, system_prompt='')
