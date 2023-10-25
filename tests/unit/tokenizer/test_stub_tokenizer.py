import pytest

from canopy.models.data_models import MessageBase, Role
from .base_test_tokenizer import BaseTestTokenizer
from ..stubs.stub_tokenizer import StubTokenizer


class TestStubTokenizer(BaseTestTokenizer):

    @staticmethod
    @pytest.fixture(scope="class")
    def tokenizer():
        return StubTokenizer()

    @staticmethod
    @pytest.fixture
    def expected_tokens(text):
        return text.split()

    @staticmethod
    def test_tokenize_detokenize_compatibility(tokenizer, text, expected_tokens):
        assert tokenizer.detokenize(tokenizer.tokenize(text)) \
               == " ".join(text.split())
        assert tokenizer.tokenize(tokenizer.detokenize(expected_tokens))\
               == expected_tokens

    @staticmethod
    def test_messages_token_count(tokenizer):
        messages = [MessageBase(role=Role.USER, content="hi bye"),
                    MessageBase(role=Role.ASSISTANT, content="hi")]
        assert tokenizer.messages_token_count(messages) == 3 + len(messages) * 3

    @staticmethod
    def test_messages_token_count_empty_messages(tokenizer):
        assert tokenizer.messages_token_count([]) == 0
