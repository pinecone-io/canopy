import os
from unittest.mock import MagicMock

import jsonschema
import pytest

from canopy.llm import AzureOpenAILLM, AnyscaleLLM
from canopy.models.data_models import Role, MessageBase, Context, StringContextContent  # noqa
from canopy.models.api_models import ChatResponse, StreamingChatChunk # noqa
from canopy.llm.openai import OpenAILLM  # noqa
from canopy.llm.models import \
    Function, FunctionParameters, FunctionArrayProperty  # noqa
from openai import BadRequestError # noqa

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


@pytest.fixture
def model_params_high_temperature():
    return {"temperature": 0.9, "top_p": 0.95, "n": 3}


@pytest.fixture
def model_params_low_temperature():
    return {"temperature": 0.2, "top_p": 0.5, "n": 1}


@pytest.fixture(params=[OpenAILLM, AzureOpenAILLM, AnyscaleLLM])
def openai_llm(request):
    llm_class = request.param
    if llm_class == AzureOpenAILLM:
        model_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        if model_name is None:
            pytest.skip(
                "Couldn't find Azure deployment name. Skipping Azure OpenAI tests."
            )
    elif llm_class == AnyscaleLLM:
        if os.getenv("ANYSCALE_API_KEY") is None:
            pytest.skip("Couldn't find Anyscale API key. Skipping Anyscale tests.")
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    else:
        model_name = "gpt-3.5-turbo-0613"

    return llm_class(model_name=model_name)


def test_init_with_custom_params(openai_llm):
    if isinstance(openai_llm, AzureOpenAILLM):
        pytest.skip("Tested separately in test_azure_openai.py")

    llm = openai_llm.__class__(
        model_name="test_model_name",
        api_key="test_api_key",
        organization="test_organization",
        temperature=0.9,
        top_p=0.95,
        n=3,
    )

    assert llm.model_name == "test_model_name"
    assert llm.default_model_params["temperature"] == 0.9
    assert llm.default_model_params["top_p"] == 0.95
    assert llm.default_model_params["n"] == 3
    assert llm._client.api_key == "test_api_key"
    assert llm._client.organization == "test_organization"


def test_chat_completion_no_context(openai_llm, messages):
    response = openai_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages)
    assert_chat_completion(response)


def test_chat_completion_with_context(openai_llm, messages):
    response = openai_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages,
                                          context=Context(
                                              content=StringContextContent(
                                                  __root__="context from kb"
                                              ),
                                              num_tokens=5
                                          ))
    assert_chat_completion(response)


def test_enforced_function_call(openai_llm,
                                messages,
                                function_query_knowledgebase):
    result = openai_llm.enforced_function_call(
        system_prompt=SYSTEM_PROMPT,
        chat_history=messages,
        function=function_query_knowledgebase)
    assert_function_call_format(result)


def test_chat_completion_high_temperature(openai_llm,
                                          messages,
                                          model_params_high_temperature):
    if isinstance(openai_llm, AnyscaleLLM):
        pytest.skip("Anyscale don't support n>1 for the moment.")

    response = openai_llm.chat_completion(
        system_prompt=SYSTEM_PROMPT,
        chat_history=messages,
        model_params=model_params_high_temperature
    )
    assert_chat_completion(response,
                           num_choices=model_params_high_temperature["n"])


def test_chat_completion_low_temperature(openai_llm,
                                         messages,
                                         model_params_low_temperature):
    response = openai_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages,
                                          model_params=model_params_low_temperature)
    assert_chat_completion(response,
                           num_choices=model_params_low_temperature["n"])


def test_enforced_function_call_high_temperature(openai_llm,
                                                 messages,
                                                 function_query_knowledgebase,
                                                 model_params_high_temperature):
    if isinstance(openai_llm, AnyscaleLLM):
        pytest.skip("Anyscale don't support n>1 for the moment.")

    result = openai_llm.enforced_function_call(
        system_prompt=SYSTEM_PROMPT,
        chat_history=messages,
        function=function_query_knowledgebase,
        model_params=model_params_high_temperature
    )
    assert isinstance(result, dict)


def test_enforced_function_call_low_temperature(openai_llm,
                                                messages,
                                                function_query_knowledgebase,
                                                model_params_low_temperature):
    model_params = model_params_low_temperature.copy()
    if isinstance(openai_llm, AnyscaleLLM):
        model_params["top_p"] = 1.0

    result = openai_llm.enforced_function_call(
        system_prompt=SYSTEM_PROMPT,
        chat_history=messages,
        function=function_query_knowledgebase,
        model_params=model_params
    )
    assert_function_call_format(result)


def test_chat_completion_with_model_name(openai_llm, messages):
    if isinstance(openai_llm, AzureOpenAILLM):
        pytest.skip("In Azure the model name has to be a valid deployment")
    elif isinstance(openai_llm, AnyscaleLLM):
        new_model_name = "meta-llama/Llama-2-7b-chat-hf"
    else:
        new_model_name = "gpt-3.5-turbo-1106"

    assert new_model_name != openai_llm.model_name, (
        "The new model name should be different from the default one. Please change it."
    )
    response = openai_llm.chat_completion(
        system_prompt=SYSTEM_PROMPT,
        chat_history=messages,
        model_params={"model": new_model_name}
    )

    assert response.model == new_model_name


def test_chat_streaming(openai_llm, messages):
    stream = True
    response = openai_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages,
                                          stream=stream)
    messages_received = [message for message in response]
    assert len(messages_received) > 0
    for message in messages_received:
        assert isinstance(message, StreamingChatChunk)


def test_max_tokens(openai_llm, messages):
    max_tokens = 2
    response = openai_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages,
                                          max_tokens=max_tokens)
    assert isinstance(response, ChatResponse)
    assert len(response.choices[0].message.content.split()) <= max_tokens


def test_negative_max_tokens(openai_llm, messages):
    with pytest.raises(RuntimeError):
        openai_llm.chat_completion(
            system_prompt=SYSTEM_PROMPT,
            chat_history=messages,
            max_tokens=-5)


def test_chat_complete_api_failure_populates(openai_llm,
                                             messages):
    openai_llm._client = MagicMock()
    openai_llm._client.chat.completions.create.side_effect = Exception(
        "API call failed")

    with pytest.raises(Exception, match="API call failed"):
        openai_llm.chat_completion(system_prompt=SYSTEM_PROMPT,
                                   chat_history=messages)


def test_enforce_function_api_failure_populates(openai_llm,
                                                messages,
                                                function_query_knowledgebase):
    openai_llm._client = MagicMock()
    openai_llm._client.chat.completions.create.side_effect = Exception(
        "API call failed")

    with pytest.raises(Exception, match="API call failed"):
        openai_llm.enforced_function_call(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages,
                                          function=function_query_knowledgebase)


def test_enforce_function_wrong_output_schema(openai_llm,
                                              messages,
                                              function_query_knowledgebase):
    openai_llm._client = MagicMock()
    openai_llm._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                tool_calls=[
                    MagicMock(
                        function=MagicMock(
                            arguments="{\"key\": \"value\"}"))]))])

    with pytest.raises(jsonschema.ValidationError,
                       match="'queries' is a required property"):
        openai_llm.enforced_function_call(system_prompt=SYSTEM_PROMPT,
                                          chat_history=messages,
                                          function=function_query_knowledgebase)

    assert openai_llm._client.chat.completions.create.call_count == 3, \
        "retry did not happen as expected"


def test_enforce_function_unsupported_model(openai_llm,
                                            messages,
                                            function_query_knowledgebase):
    if isinstance(openai_llm, AzureOpenAILLM):
        pytest.skip("Currently not tested in Azure")
    elif isinstance(openai_llm, AnyscaleLLM):
        new_model_name = "meta-llama/Llama-2-7b-chat-hf"
    else:
        new_model_name = "gpt-3.5-turbo-0301"

    with pytest.raises(NotImplementedError):
        openai_llm.enforced_function_call(
            system_prompt=SYSTEM_PROMPT,
            chat_history=messages,
            function=function_query_knowledgebase,
            model_params={"model": new_model_name}
        )


def test_available_models(openai_llm):
    if isinstance(openai_llm, AzureOpenAILLM):
        pytest.skip("Azure does not support listing models")
    models = openai_llm.available_models
    assert isinstance(models, list)
    assert len(models) > 0
    assert isinstance(models[0], str)
    assert openai_llm.model_name in models


@pytest.fixture()
def no_api_key():
    before = os.environ.pop("OPENAI_API_KEY", None)
    yield
    if before is not None:
        os.environ["OPENAI_API_KEY"] = before


def test_missing_api_key(no_api_key):
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAILLM()


@pytest.fixture()
def bad_api_key():
    before = os.environ.pop("OPENAI_API_KEY", None)
    os.environ["OPENAI_API_KEY"] = "bad key"
    yield
    if before is not None:
        os.environ["OPENAI_API_KEY"] = before


def test_bad_api_key(bad_api_key, messages):
    with pytest.raises(RuntimeError, match="API key"):
        llm = OpenAILLM()
        llm.chat_completion(system_prompt=SYSTEM_PROMPT, chat_history=messages)
