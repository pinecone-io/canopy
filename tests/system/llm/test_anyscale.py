from unittest.mock import MagicMock

import pytest


from canopy.models.data_models import Role, MessageBase, Context, StringContextContent  # noqa
from canopy.models.api_models import ChatResponse, StreamingChatChunk  # noqa
from canopy.llm.anyscale import AnyscaleLLM  # noqa
from canopy.llm.models import (
    Function,
    FunctionParameters,
    FunctionArrayProperty,
)  # noqa
from openai import BadRequestError  # noqa


SYSTEM_PROMPT = "You are a helpful assistant."


def assert_chat_completion(response, num_choices=1):
    assert len(response.choices) == num_choices
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


class TestAnyscaleLLM:
    @staticmethod
    @pytest.fixture
    def model_name():
        return "mistralai/Mistral-7B-Instruct-v0.1"

    @staticmethod
    @pytest.fixture
    def messages():
        # Create a list of MessageBase objects
        return [
            MessageBase(role=Role.USER, content="Hello, assistant. "),
        ]

    @staticmethod
    @pytest.fixture
    def function_query_knowledgebase():
        return Function(
            name="query_knowledgebase",
            description="Query search engine for relevant information",
            parameters=FunctionParameters(
                required_properties=[
                    FunctionArrayProperty(
                        name="queries",
                        items_type="string",
                        description="List of queries to send to the search engine.",
                    ),
                ]
            ),
        )

    @staticmethod
    @pytest.fixture
    def model_params_high_temperature():
        # `n` parameter is not supported yet. set to 1 always
        return {"temperature": 0.9, "n": 1}

    @staticmethod
    @pytest.fixture
    def model_params_low_temperature():
        return {"temperature": 0.2, "n": 1}

    @staticmethod
    @pytest.fixture
    def anyscale_llm(model_name):
        return AnyscaleLLM(model_name=model_name)

    @staticmethod
    def test_init_with_custom_params(anyscale_llm):
        llm = AnyscaleLLM(
            model_name="test_model_name",
            api_key="test_api_key",
            temperature=0.9,
            top_p=0.95,
            n=3,
        )

        assert llm.model_name == "test_model_name"
        assert llm.default_model_params["temperature"] == 0.9
        assert llm.default_model_params["top_p"] == 0.95
        assert llm.default_model_params["n"] == 3
        assert llm._client.api_key == "test_api_key"

    @staticmethod
    def test_chat_completion(anyscale_llm, messages):
        response = anyscale_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                                chat_history=messages)
        assert_chat_completion(response)

    @staticmethod
    def test_enforced_function_call(
        anyscale_llm, messages, function_query_knowledgebase
    ):
        result = anyscale_llm.enforced_function_call(
            system_prompt=SYSTEM_PROMPT, chat_history=messages,
            function=function_query_knowledgebase
        )
        assert_function_call_format(result)

    @staticmethod
    def test_chat_completion_high_temperature(
        anyscale_llm, messages, model_params_high_temperature
    ):
        response = anyscale_llm.chat_completion(
            system_prompt=SYSTEM_PROMPT, chat_history=messages,
            model_params=model_params_high_temperature
        )
        assert_chat_completion(response, num_choices=model_params_high_temperature["n"])

    @staticmethod
    def test_chat_completion_low_temperature(
        anyscale_llm, messages, model_params_low_temperature
    ):
        response = anyscale_llm.chat_completion(
            system_prompt=SYSTEM_PROMPT, chat_history=messages,
            model_params=model_params_low_temperature
        )
        assert_chat_completion(response, num_choices=model_params_low_temperature["n"])

    @staticmethod
    def test_enforced_function_call_high_temperature(
        anyscale_llm,
        messages,
        function_query_knowledgebase,
        model_params_high_temperature,
    ):
        result = anyscale_llm.enforced_function_call(
            system_prompt=SYSTEM_PROMPT,
            chat_history=messages,
            function=function_query_knowledgebase,
            model_params=model_params_high_temperature,
        )
        assert_function_call_format(result)

    @staticmethod
    def test_enforced_function_call_low_temperature(
        anyscale_llm,
        messages,
        function_query_knowledgebase,
        model_params_low_temperature,
    ):
        result = anyscale_llm.enforced_function_call(
            system_prompt=SYSTEM_PROMPT,
            chat_history=messages,
            function=function_query_knowledgebase,
            model_params=model_params_low_temperature,
        )
        assert_function_call_format(result)

    @staticmethod
    def test_chat_streaming(anyscale_llm, messages):
        stream = True
        response = anyscale_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                                chat_history=messages, stream=stream)
        messages_received = [message for message in response]
        assert len(messages_received) > 0
        for message in messages_received:
            assert isinstance(message, StreamingChatChunk)

    @staticmethod
    def test_max_tokens(anyscale_llm, messages):
        max_tokens = 2
        response = anyscale_llm.chat_completion(
            system_prompt=SYSTEM_PROMPT, chat_history=messages, max_tokens=max_tokens
        )
        assert isinstance(response, ChatResponse)
        assert len(response.choices[0].message.content.split()) <= max_tokens

    @staticmethod
    @pytest.mark.skip("For now AE does not throw exceptions for missing messages")
    def test_missing_messages(anyscale_llm):
        with pytest.raises(BadRequestError):
            anyscale_llm.chat_completion(messages=[])

    @staticmethod
    def test_negative_max_tokens(anyscale_llm, messages):
        with pytest.raises(RuntimeError):
            anyscale_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                         chat_history=messages, max_tokens=-5)

    @staticmethod
    def test_chat_complete_api_failure_populates(anyscale_llm, messages):
        anyscale_llm._client = MagicMock()
        anyscale_llm._client.chat.completions.create.side_effect = Exception(
            "API call failed"
        )

        with pytest.raises(Exception, match="API call failed"):
            anyscale_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                         chat_history=messages)

    @staticmethod
    def test_available_models(anyscale_llm):
        models = anyscale_llm.available_models
        assert isinstance(models, list)
        assert len(models) > 0
        assert isinstance(models[0], str)
        assert anyscale_llm.model_name in models
