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
        ]

    @staticmethod
    def test_messages_token_count(tokenizer):
        messages = [MessageBase(role=Role.USER, content="Hello, assistant.")]
        assert tokenizer.messages_token_count(messages) == 11

        messages = [
            MessageBase(role=Role.USER, content="Hello, assistant."),
            MessageBase(
                role=Role.ASSISTANT, content="Hello, user. How can I assist you?"
            ),
        ]
        assert tokenizer.messages_token_count(messages) == 25

    @staticmethod
    def test_special_tokens_to_natural_text(tokenizer):
        tokens = tokenizer.tokenize("<BOS_TOKEN>")
        assert tokens == ['<', 'BOS', '_', 'TOKEN', '>']

        assert tokenizer.detokenize(tokens) == "<BOS_TOKEN>"

        assert tokenizer.token_count("<BOS_TOKEN>") == 5
