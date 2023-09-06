from unittest.mock import patch
import pytest


from context_engine.models.data_models import Role, MessageBase # noqa
from context_engine.models.api_models import ChatResponse, StreamingChatResponse # noqa
from context_engine.llm.openai import OpenAILLM # noqa
from context_engine.llm.models import \
    Function, FunctionParameters, FunctionArrayProperty, ModelParams # noqa
from openai import InvalidRequestError # noqa


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


class TestOpenAILLM:

    @staticmethod
    @pytest.fixture
    def model_name():
        return "gpt-3.5-turbo-0613"

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
    def function_query_knowledgebase():
        return Function(
            name="query_knowledgebase",
            description="Query search engine for relevant information",
            parameters=FunctionParameters(
                required_properties=[
                    FunctionArrayProperty(
                        name="queries",
                        items_type="string",
                        description='List of queries to send to the search engine.',
                    ),
                ]
            ),
        )

    @staticmethod
    @pytest.fixture
    def model_params_high_temperature():
        return ModelParams(temperature=0.9, top_p=0.95, n=3)

    @staticmethod
    @pytest.fixture
    def model_params_low_temperature():
        return ModelParams(temperature=0.2, top_p=0.5, n=1)

    @staticmethod
    @pytest.fixture
    def openai_llm(model_name):
        return OpenAILLM(model_name=model_name)

    @staticmethod
    def test_chat_completion(openai_llm, messages):
        response = openai_llm.chat_completion(messages=messages)
        assert_chat_completion(response)

    @staticmethod
    def test_enforced_function_call(openai_llm,
                                    messages,
                                    function_query_knowledgebase):
        result = openai_llm.enforced_function_call(
            messages=messages,
            function=function_query_knowledgebase)
        assert_function_call_format(result)

    @staticmethod
    def test_chat_completion_high_temperature(openai_llm,
                                              messages,
                                              model_params_high_temperature):
        response = openai_llm.chat_completion(
            messages=messages,
            model_params=model_params_high_temperature
        )
        assert_chat_completion(response,
                               num_choices=model_params_high_temperature.n)

    @staticmethod
    def test_chat_completion_low_temperature(openai_llm,
                                             messages,
                                             model_params_low_temperature):
        response = openai_llm.chat_completion(messages=messages,
                                              model_params=model_params_low_temperature)
        assert_chat_completion(response,
                               num_choices=model_params_low_temperature.n)

    @staticmethod
    def test_enforced_function_call_high_temperature(openai_llm,
                                                     messages,
                                                     function_query_knowledgebase,
                                                     model_params_high_temperature):
        result = openai_llm.enforced_function_call(
            messages=messages,
            function=function_query_knowledgebase,
            model_params=model_params_high_temperature
        )
        assert isinstance(result, dict)

    @staticmethod
    def test_enforced_function_call_low_temperature(openai_llm,
                                                    messages,
                                                    function_query_knowledgebase,
                                                    model_params_low_temperature):
        result = openai_llm.enforced_function_call(
            messages=messages,
            function=function_query_knowledgebase,
            model_params=model_params_low_temperature
        )
        assert_function_call_format(result)

    @staticmethod
    def test_chat_streaming(openai_llm, messages):
        stream = True
        response = openai_llm.chat_completion(messages=messages,
                                              stream=stream)
        messages_received = [message for message in response]
        assert len(messages_received) > 0
        for message in messages_received:
            assert isinstance(message, StreamingChatResponse)

    @staticmethod
    def test_max_tokens(openai_llm, messages):
        max_tokens = 2
        response = openai_llm.chat_completion(messages=messages,
                                              max_tokens=max_tokens)
        assert isinstance(response, ChatResponse)
        assert len(response.choices[0].message.content.split()) <= max_tokens

    @staticmethod
    def test_invalid_model_name():
        with pytest.raises(ValueError, match="Model invalid_model_name not found."):
            OpenAILLM(model_name="invalid_model_name")

    @staticmethod
    def test_missing_messages(openai_llm):
        with pytest.raises(InvalidRequestError):
            openai_llm.chat_completion(messages=[])

    @staticmethod
    def test_negative_max_tokens(openai_llm, messages):
        with pytest.raises(InvalidRequestError):
            openai_llm.chat_completion(messages=messages, max_tokens=-5)

    @staticmethod
    @patch("openai.ChatCompletion.create")
    def test_chat_complete_api_failure_populates(mock_api_call,
                                                 openai_llm,
                                                 messages):
        mock_api_call.side_effect = Exception("API call failed")

        with pytest.raises(Exception, match="API call failed"):
            openai_llm.chat_completion(messages=messages)

    @staticmethod
    @patch("openai.ChatCompletion.create")
    def test_enforce_function_api_failure_populates(mock_api_call,
                                                    openai_llm,
                                                    messages,
                                                    function_query_knowledgebase):
        mock_api_call.side_effect = Exception("API call failed")

        with pytest.raises(Exception, match="API call failed"):
            openai_llm.enforced_function_call(messages=messages,
                                              function=function_query_knowledgebase)
