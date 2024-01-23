import pytest

from canopy.tokenizer import CohereAPITokenizer


class TestCohereAPITokenizer:
    @staticmethod
    @pytest.fixture(scope="class")
    def tokenizer():
        return CohereAPITokenizer(model_name="command")

    @staticmethod
    @pytest.fixture
    def text():
        return "Hello World!"

    @staticmethod
    @pytest.fixture
    def expected_tokens(text):
        return ["Hello", " World", "!"]

    # region: test tokenize

    @staticmethod
    def test_tokenize(tokenizer, text, expected_tokens):
        tokens = tokenizer.tokenize(text)
        assert tokens == expected_tokens, f"\nExpected: {expected_tokens}" \
                                          f"\nActual: {tokens}"

    @staticmethod
    def test_tokenize_empty_string(tokenizer):
        assert tokenizer.tokenize("") == []

    @staticmethod
    def test_tokenize_invalid_input_type_raise_exception(tokenizer):
        with pytest.raises(Exception):
            tokenizer.tokenize(1)

        with pytest.raises(Exception):
            tokenizer.tokenize(["asd"])

    # endregion

    # region: test detokenize

    @staticmethod
    def test_detokenize(tokenizer, text, expected_tokens):
        text = tokenizer.detokenize(expected_tokens)
        assert text == text

    @staticmethod
    def test_detokenize_empty_string(tokenizer):
        assert tokenizer.detokenize([]) == ""

    @staticmethod
    def test_detokenize_invalid_input_type_raise_exception(tokenizer):
        with pytest.raises(Exception):
            tokenizer.detokenize(1)

        with pytest.raises(Exception):
            tokenizer.detokenize("asd")

    # region test token_count

    @staticmethod
    def test_token_count(tokenizer, text, expected_tokens):
        token_count = tokenizer.token_count(text)
        assert token_count == len(expected_tokens)
        assert token_count == len(tokenizer.tokenize(text))

    @staticmethod
    def test_token_count_empty_string(tokenizer):
        assert tokenizer.token_count("") == 0

    @staticmethod
    def test_messages_token_count_empty_messages(tokenizer):
        assert tokenizer.messages_token_count([]) == 3

    # endregion

    # region special tokens

    @staticmethod
    def test_special_tokens_to_natural_text(tokenizer):
        tokens = tokenizer.tokenize("<|endoftext|>")

        assert tokens == ['<', '|', 'end', 'of', 'text', '|', '>']
        assert tokenizer.detokenize(tokens) == "<|endoftext|>"
        assert tokenizer.token_count("<|endoftext|>") == 7

    # endregion
