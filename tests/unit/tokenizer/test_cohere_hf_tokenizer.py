import pytest
from canopy.tokenizer import CohereHFTokenizer
from canopy.models.data_models import MessageBase, Role
from .base_test_tokenizer import BaseTestTokenizer


class TestCohereHFTokenizer(BaseTestTokenizer):
    @staticmethod
    @pytest.fixture(scope="class")
    def tokenizer():
        return CohereHFTokenizer()

    @staticmethod
    @pytest.fixture
    def expected_tokens(text):
        return [
            '<BOS_TOKEN>',
            'string',
            'Ġwith',
            'Ġspecial',
            'Ġcharacters',
            'Ġlike',
            'Ġ!',
            '@',
            '#',
            '$',
            '%',
            '^',
            '&',
            '*',
            '()',
            '_',
            '+',
            'Ġæ',
            'Ĺ',
            '¥',
            'æľ¬',
            'Ġspaces',
            'ĠĠĠ',
            'ĊĠ',
            'ĊĊ',
            'ĠCASE',
            'Ġc',
            'A',
            'se',
            'Ġ',
            '<EOP_TOKEN>'
        ]

    @staticmethod
    def test_tokenize_empty_string(tokenizer):
        """
        Overrides test from base class because Cohere considers that
        an empty string has two tokens, not zero.
        """
        assert tokenizer.tokenize("") == ['<BOS_TOKEN>', '<EOP_TOKEN>']

    @staticmethod
    def test_token_count_empty_string(tokenizer):
        """
        Overrides test from base class because Cohere considers that
        an empty string has two tokens, not zero.
        """
        assert tokenizer.token_count("") == 2

    @staticmethod
    def test_messages_token_count(tokenizer):
        messages = [MessageBase(role=Role.USER, content="Hello, assistant.")]
        assert tokenizer.messages_token_count(messages) == 15

        messages = [
            MessageBase(role=Role.USER, content="Hello, assistant."),
            MessageBase(
                role=Role.ASSISTANT, content="Hello, user. How can I assist you?"
            ),
        ]
        assert tokenizer.messages_token_count(messages) == 33

    @staticmethod
    def test_messages_token_count_empty_messages(tokenizer):
        assert tokenizer.messages_token_count([]) == 3

    @staticmethod
    def test_special_tokens_to_natural_text(tokenizer):
        input_text = "</s>_<0x0A>__ <unk><s>word"
        tokens = tokenizer.tokenize(input_text)

        expected_tokens = [
            '<BOS_TOKEN>',
            '</',
            's',
            '>',
            '_<',
            '0',
            'x',
            '0',
            'A',
            '>',
            '__',
            'Ġ<',
            'unk',
            '><',
            's',
            '>',
            'word',
            '<EOP_TOKEN>',
        ]

        assert tokens == expected_tokens
        assert tokenizer.detokenize(tokens) == input_text
        assert tokenizer.token_count(input_text) == len(tokens)
