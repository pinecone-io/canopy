import os

import pytest

from canopy.llm import AzureOpenAILLM
from .test_openai import SYSTEM_PROMPT

MODEL_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")


@pytest.fixture
def azure_openai_llm():
    if os.getenv("AZURE_DEPLOYMENT_NAME") is None:
        pytest.skip(
            "Couldn't find Azure deployment name. Skipping Azure OpenAI tests."
        )
    return AzureOpenAILLM(model_name=os.getenv("AZURE_DEPLOYMENT_NAME"))


def test_init_params(azure_openai_llm):
    llm = AzureOpenAILLM(
        model_name="test_model_name",
        api_version="2020-05-03",
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
    assert llm._client._api_version == "2020-05-03"


@pytest.fixture()
def no_api_key():
    before = os.environ.pop("AZURE_OPENAI_API_KEY", None)
    yield
    if before is not None:
        os.environ["AZURE_OPENAI_API_KEY"] = before


def test_missing_api_key(no_api_key):
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY"):
        AzureOpenAILLM(MODEL_NAME)


@pytest.fixture()
def bad_api_key():
    before = os.environ.pop("AZURE_OPENAI_API_KEY", None)
    os.environ["AZURE_OPENAI_API_KEY"] = "bad key"
    yield
    if before is not None:
        os.environ["AZURE_OPENAI_API_KEY"] = before


def test_bad_api_key(bad_api_key, messages):
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY"):
        llm = AzureOpenAILLM(MODEL_NAME)
        llm.chat_completion(system_prompt=SYSTEM_PROMPT, chat_history=messages)


@pytest.fixture()
def no_azure_endpoint():
    before = os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
    yield
    if before is not None:
        os.environ["AZURE_OPENAI_ENDPOINT"] = before


def test_missing_azure_endpoint(no_azure_endpoint):
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
        AzureOpenAILLM(MODEL_NAME)


@pytest.fixture()
def bad_azure_endpoint():
    before = os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
    os.environ["AZURE_OPENAI_ENDPOINT"] = "bad endpoint"
    yield
    if before is not None:
        os.environ["AZURE_OPENAI_ENDPOINT"] = before


def test_bad_azure_endpoint(bad_azure_endpoint, messages):
    with pytest.raises(RuntimeError, match="Azure OpenAI endpoint"):
        llm = AzureOpenAILLM(MODEL_NAME)
        llm.chat_completion(system_prompt=SYSTEM_PROMPT, chat_history=messages)

# def test_function_calling_error(azure_openai_llm):
