import os

import pytest

from canopy.models.data_models import MessageBase, Role
from canopy.tokenizer import CohereAPITokenizer
from tokenizer.base_test_tokenizer import BaseTestTokenizer


class TestCohereAPITokenizer(BaseTestTokenizer):
    @staticmethod
    @pytest.fixture(scope="class")
    def tokenizer():
        if not os.getenv("CO_API_KEY"):
            pytest.skip("Skipping Cohere API tokenizer tests because "
                        "COHERE_API_KEY environment variable is not set.")
        return CohereAPITokenizer(model_name="command")

    @staticmethod
    @pytest.fixture
    def text():
        return "string with special characters like !@#$%^&*()_+日本 " \
               "spaces   \n \n\n CASE cAse "

    @staticmethod
    @pytest.fixture
    def expected_tokens(text):
        return ['string', ' with', ' special', ' characters', ' like',
                ' !', '@', '#', '$', '%', '^', '&', '*', '()', '_', '+', '日',
                '本', ' spaces', '   ', '\n ', '\n\n', ' CASE', ' c', 'A',
                'se', " "]

    @staticmethod
    def test_messages_token_count(tokenizer):
        messages = [MessageBase(role=Role.USER, content="Hello, assistant.")]
        assert tokenizer.messages_token_count(messages) == 11

        messages = [MessageBase(role=Role.USER,
                                content="Hello, assistant."),
                    MessageBase(role=Role.ASSISTANT,
                                content="Hello, user. How can I assist you?")]
        assert tokenizer.messages_token_count(messages) == 25

    @staticmethod
    def test_messages_token_count_empty_messages(tokenizer):
        assert tokenizer.messages_token_count([]) == 3
