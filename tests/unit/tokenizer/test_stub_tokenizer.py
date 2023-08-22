import pytest

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
