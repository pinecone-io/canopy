import os
from typing import List

from datetime import datetime

import pinecone
import pytest
from fastapi.testclient import TestClient
from tenacity import retry, stop_after_attempt, wait_fixed

from canopy.knowledge_base import KnowledgeBase

from canopy_cli.app import app
from canopy_cli.api_models import HealthStatus, ContextUpsertRequest, ContextQueryRequest
from .. import Tokenizer

upsert_payload = ContextUpsertRequest(
    documents=[
        {
            "id": "api_tests-1",
            "text": "This is a test document, the topic is red bananas",
            "source": "api_tests",
            "metadata": {"test": "test"},
        }
    ],
)


@retry(stop=stop_after_attempt(60), wait=wait_fixed(1))
def assert_vector_ids_exist(vector_ids: List[str],
                            knowledge_base: KnowledgeBase):
    fetch_response = knowledge_base._index.fetch(ids=vector_ids)
    assert all([v_id in fetch_response["vectors"] for v_id in vector_ids])


@retry(stop=stop_after_attempt(60), wait=wait_fixed(1))
def assert_vector_ids_not_exist(vector_ids: List[str],
                                knowledge_base: KnowledgeBase):
    fetch_response = knowledge_base._index.fetch(ids=vector_ids)
    assert len(fetch_response["vectors"]) == 0


@pytest.fixture(scope="module")
def index_name(testrun_uid):
    today = datetime.today().strftime("%Y-%m-%d")
    return f"test-app-{testrun_uid[-6:]}-{today}"


@pytest.fixture(scope="module", autouse=True)
def knowledge_base(index_name):
    pinecone.init()
    kb = KnowledgeBase(index_name=index_name)
    kb.create_canopy_index()

    return kb


@pytest.fixture(scope="module")
def client(knowledge_base, index_name):
    index_name_before = os.getenv("INDEX_NAME")
    os.environ["INDEX_NAME"] = index_name
    Tokenizer.clear()
    with TestClient(app) as client:
        yield client
    if index_name_before:
        os.environ["INDEX_NAME"] = index_name_before
    else:
        os.unsetenv("INDEX_NAME")


@pytest.fixture(scope="module", autouse=True)
def teardown_knowledge_base(knowledge_base):
    yield

    pinecone.init()
    index_name = knowledge_base.index_name
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)

# TODO: the following test is a complete e2e test, this it not the final design
# for the e2e tests, however there were some issues
# with the fixtures that will be resovled


def test_health(client):
    health_response = client.get("/health")
    assert health_response.is_success
    assert (
        health_response.json()
        == HealthStatus(pinecone_status="OK", llm_status="OK").dict()
    )


def test_upsert(client):
    # Upsert a document to the index
    upsert_response = client.post("/context/upsert", json=upsert_payload.dict())
    assert upsert_response.is_success


@retry(stop=stop_after_attempt(60), wait=wait_fixed(1))
def test_query(client):
    # fetch the context with all the right filters
    query_payload = ContextQueryRequest(
        queries=[
            {
                "text": "red bananas",
                "metadata_filter": {"test": "test"},
                "top_k": 1,
            }
        ],
        max_tokens=100,
    )

    query_response = client.post("/context/query", json=query_payload.dict())
    assert query_response.is_success

    # test response is as expected on /query
    response_as_json = query_response.json()

    assert (
            response_as_json[0]["query"]
            == query_payload.dict()["queries"][0]["text"]
            and response_as_json[0]["snippets"][0]["text"]
            == upsert_payload.dict()["documents"][0]["text"]
    )
    assert (response_as_json[0]["snippets"][0]["source"] ==
            upsert_payload.dict()["documents"][0]["source"])


def test_chat(client):
    # test response is as expected on /chat
    chat_payload = {
        "messages": [
            {
                "role": "user",
                "content": "what is the topic of the test document? be concise",
            }
        ]
    }
    chat_response = client.post("/context/chat/completions", json=chat_payload)
    assert chat_response.is_success
    chat_response_as_json = chat_response.json()
    assert chat_response_as_json["choices"][0]["message"]["role"] == "assistant"
    chat_response_content = chat_response_as_json["choices"][0]["message"][
        "content"
    ]
    print(chat_response_content)
    assert all([kw in chat_response_content for kw in ["red", "bananas"]])


def test_delete(client, knowledge_base):
    doc_ids = ["api_tests-1"]
    vector_ids = [f"{d_id}_{0}" for d_id in doc_ids]

    assert_vector_ids_exist(vector_ids, knowledge_base)

    delete_payload = {
        "document_ids": doc_ids
    }
    delete_response = client.post("/context/delete", json=delete_payload)
    assert delete_response.is_success

    assert_vector_ids_not_exist(vector_ids, knowledge_base)
