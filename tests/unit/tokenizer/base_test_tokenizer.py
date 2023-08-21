from abc import ABC, abstractmethod
import pytest


class BaseTestTokenizer(ABC):

    @staticmethod
    @pytest.fixture
    @abstractmethod
    def tokenizer():
        pass

    @staticmethod
    @pytest.fixture
    def text():
        return "string with special characters like !@#$%^&*()_+ 日本 " \
               "spaces   \n \n\n CASE cAse "

    @staticmethod
    @pytest.fixture
    @abstractmethod
    def expected_tokens(text):
        pass

    @staticmethod
    def test_tokenize(tokenizer, text, expected_tokens):
        tokens = tokenizer.tokenize(text)
        assert tokens == expected_tokens

    @staticmethod
    def test_tokenize_empty_string(tokenizer):
        assert tokenizer.tokenize("") == []

    @staticmethod
    def test_detokenize(tokenizer, text, expected_tokens):
        text = tokenizer.detokenize(expected_tokens)
        assert text == text

    @staticmethod
    def test_detokenize_empty_string(tokenizer):
        assert tokenizer.detokenize([]) == ""

    @staticmethod
    def test_token_count(tokenizer, text, expected_tokens):
        token_count = tokenizer.token_count(text)
        assert token_count == len(expected_tokens)

    @staticmethod
    def test_token_count_empty_string(tokenizer):
        assert tokenizer.token_count("") == 0

    @staticmethod
    def test_tokenize_detokenize_compatibility(tokenizer, text, expected_tokens):
        assert tokenizer.detokenize(tokenizer.tokenize(text)) \
               == text
        assert tokenizer.tokenize(tokenizer.detokenize(expected_tokens))\
               == expected_tokens

    @staticmethod
    def test_tokenize_invalid_input_type_raise_exception(tokenizer):
        with pytest.raises(Exception):
            tokenizer.tokenize(1)

        with pytest.raises(Exception):
            tokenizer.tokenize(["asd"])

    @staticmethod
    def test_detokenize_invalid_input_type_raise_exception(tokenizer):
        with pytest.raises(Exception):
            tokenizer.detokenize(1)

        with pytest.raises(Exception):
            tokenizer.detokenize("asd")
