import pytest
from context_engine.knoweldge_base.tokenizer import OpenAITokenizer
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
