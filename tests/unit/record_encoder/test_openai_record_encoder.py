import pytest
from pinecone_text.dense.openai_encoder import OpenAIEncoder

from canopy.knowledge_base.record_encoder.openai import OpenAIRecordEncoder
from .base_test_record_encoder import BaseTestRecordEncoder
from unittest.mock import Mock


def generate_mock_openai_response(input, model):
    """
    Generates a mock response based on the input texts.
    In a real case, this would be determined by the openai library,
    but for testing purposes, we just return a stubbed response.
    """
    assert model == "text-embedding-ada-002"
    assert isinstance(input, list)
    assert all(isinstance(text, str) for text in input)
    return {
        "data": [{"embedding": [len(text) / 100 for _ in range(1536)]}
                 for text in input]
    }


@pytest.fixture(autouse=True)
def mock_openai(monkeypatch):
    mock_create = Mock(side_effect=generate_mock_openai_response)
    monkeypatch.setattr('openai.OpenAI.embeddings.create', mock_create)


class TestOpenAIRecordEncoder(BaseTestRecordEncoder):

    @staticmethod
    @pytest.fixture
    def expected_dimension():
        return 1536

    @staticmethod
    @pytest.fixture
    def inner_encoder(expected_dimension):
        return OpenAIEncoder()

    @staticmethod
    @pytest.fixture
    def record_encoder(inner_encoder):
        return OpenAIRecordEncoder(batch_size=2)
