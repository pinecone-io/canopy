import pytest
from context_engine.knoweldge_base.tokenizer import OpenAITokenizer
from context_engine.models.data_models import MessageBase, Role
from .base_test_tokenizer import BaseTestTokenizer


class TestOpenAITokenizer(BaseTestTokenizer):

    @staticmethod
    @pytest.fixture(scope="class")
    def tokenizer():
        return OpenAITokenizer(model_name="gpt-3.5-turbo")

    @staticmethod
    @pytest.fixture
    def expected_tokens(text):
        return ['string', ' with', ' special', ' characters', ' like',
                ' !', '@', '#$', '%^', '&', '*', '()', '_', '+', ' 日',
                '本', ' spaces', '   \n', ' \n\n', ' CASE', ' c', 'A',
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
