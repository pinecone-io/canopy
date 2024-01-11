from unittest.mock import MagicMock

import pytest


from canopy.models.data_models import Role, MessageBase, Context, StringContextContent  # noqa
from canopy.models.api_models import ChatResponse, StreamingChatChunk  # noqa
from canopy.llm.anyscale import AnyscaleLLM  # noqa
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


@pytest.fixture
def model_name():
    return "meta-llama/Llama-2-7b-chat-hf"


@pytest.fixture
def messages():
    # Create a list of MessageBase objects
    return [
        MessageBase(role=Role.USER, content="Hello, assistant."),
        MessageBase(
            role=Role.ASSISTANT, content="Hello, user. How can I assist you?"
        ),
    ]


@pytest.fixture
def model_params_high_temperature():
    # `n` parameter is not supported yet. set to 1 always
    return {"temperature": 0.9, "top_p": 0.95, "n": 1}


@pytest.fixture
def model_params_low_temperature():
    return {"temperature": 0.2, "top_p": 0.5, "n": 1}


@pytest.fixture
def anyscale_llm(model_name):
    return AnyscaleLLM(model_name=model_name)


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


def test_chat_completion(anyscale_llm, messages):
    response = anyscale_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                            chat_history=messages)
    assert_chat_completion(response)


def test_chat_completion_with_context(anyscale_llm, messages):
    response = anyscale_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                            chat_history=messages,
                                            context=Context(
                                                content=StringContextContent(
                                                    __root__="context from kb"
                                                ),
                                                num_tokens=5)
                                            )
    assert_chat_completion(response)


def test_chat_completion_high_temperature(
    anyscale_llm, messages, model_params_high_temperature
):
    response = anyscale_llm.chat_completion(
        system_prompt=SYSTEM_PROMPT,
        chat_history=messages,
        model_params=model_params_high_temperature
    )
    assert_chat_completion(response, num_choices=model_params_high_temperature["n"])


def test_chat_completion_low_temperature(
    anyscale_llm, messages, model_params_low_temperature
):
    response = anyscale_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                            chat_history=messages,
                                            model_params=model_params_low_temperature)
    assert_chat_completion(response, num_choices=model_params_low_temperature["n"])


def test_chat_streaming(anyscale_llm, messages):
    stream = True
    response = anyscale_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                            chat_history=messages,
                                            stream=stream)
    messages_received = [message for message in response]
    assert len(messages_received) > 0
    for message in messages_received:
        assert isinstance(message, StreamingChatChunk)


def test_max_tokens(anyscale_llm, messages):
    max_tokens = 2
    response = anyscale_llm.chat_completion(
        system_prompt=SYSTEM_PROMPT,
        chat_history=messages,
        max_tokens=max_tokens
    )
    assert isinstance(response, ChatResponse)
    assert len(response.choices[0].message.content.split()) <= max_tokens


def test_negative_max_tokens(anyscale_llm, messages):
    with pytest.raises(RuntimeError):
        anyscale_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                     chat_history=messages,
                                     max_tokens=-5)


def test_chat_complete_api_failure_populates(anyscale_llm, messages):
    anyscale_llm._client = MagicMock()
    anyscale_llm._client.chat.completions.create.side_effect = Exception(
        "API call failed"
    )

    with pytest.raises(Exception, match="API call failed"):
        anyscale_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                     chat_history=messages)


def test_available_models(anyscale_llm):
    models = anyscale_llm.available_models
    assert isinstance(models, list)
    assert len(models) > 0
    assert isinstance(models[0], str)
    assert anyscale_llm.model_name in models
