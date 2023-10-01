import os
from fastapi.testclient import TestClient
from tenacity import retry, stop_after_attempt, wait_fixed

from resin.knoweldge_base import KnowledgeBase
from resin.knoweldge_base.tokenizer import OpenAITokenizer, Tokenizer

from resin_cli.app import app, _init_engines, _init_logging
from resin_cli.api_models import HealthStatus, ContextUpsertRequest, ContextQueryRequest


client = TestClient(app)

# TODO: the following test is a complete e2e test, this it not the final design
# for the e2e tests, however there were some issues
# with the fixtures that will be resovled


def test_e2e():
    _init_logging()
    Tokenizer.initialize(OpenAITokenizer, "gpt-3.5-turbo")
    kb = KnowledgeBase.create_with_new_index(
        index_name=os.environ["INDEX_NAME"],
        encoder=KnowledgeBase.DEFAULT_RECORD_ENCODER(),
        chunker=KnowledgeBase.DEFAULT_CHUNKER(),
    )
    Tokenizer.clear()
    _init_engines()

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert (
        health_response.json()
        == HealthStatus(pinecone_status="OK", llm_status="OK").dict()
    )

    try:
        # Upsert a document to the index
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
        upsert_response = client.post("/context/upsert", json=upsert_payload.dict())
        assert upsert_response.status_code == 200

        # fetech the context with all the right filters
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
        assert query_response.status_code == 200

        # test response is as expected on /query
        response_as_json = query_response.json()

        @retry(stop=stop_after_attempt(60), wait=wait_fixed(1))
        def test_response_is_expected(response_as_json):
            (
                response_as_json[0]["query"]
                == query_payload.dict()["queries"][0]["text"]
                and response_as_json[0]["snippets"][0]["text"]
                == upsert_payload.dict()["documents"][0]["text"]
            )

        assert test_response_is_expected(response_as_json)

        # TODO: uncomment when fix is pushed
        # assert response_as_json[0]["snippets"][0]["source"] == \
        # upsert_payload.dict()["documents"][0]["source"]

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
        assert chat_response.status_code == 200
        chat_response_as_json = chat_response.json()
        assert chat_response_as_json["choices"][0]["message"]["role"] == "assistant"
        chat_response_content = chat_response_as_json["choices"][0]["message"][
            "content"
        ]
        print(chat_response_content)
        assert all([kw in chat_response_content for kw in ["red", "bananas"]])
    except Exception as e:
        raise e
    finally:
        kb.delete_index()
