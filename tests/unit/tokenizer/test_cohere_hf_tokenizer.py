import pytest
from canopy.tokenizer import CohereHFTokenizer
from canopy.models.data_models import MessageBase, Role
from .base_test_tokenizer import BaseTestTokenizer


class TestCohereHFTokenizer(BaseTestTokenizer):
    @staticmethod
    @pytest.fixture(scope="class")
    def tokenizer():
        try:
            tokenizer = CohereHFTokenizer()
        except ImportError:
            pytest.skip(
                "`cohere` extra not installed. Skipping CohereHFTokenizer unit "
                "tests"
            )
        return tokenizer

    @staticmethod
    @pytest.fixture
    def expected_tokens(text):
        return ['string',
                'Ġwith',
                'Ġspecial',
                'Ġcharacters',
                'Ġlike',
                'Ġ!',
                '@',
                '#$',
                '%^',
                '&',
                '*',
                '()',
                '_',
                '+',
                'ĠæĹ¥æľ¬',
                'Ġspaces',
                'ĠĠĠ',
                'ĊĠĊĊ',
                'ĠCASE',
                'Ġc',
                'A',
                'se',
                'Ġ']

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
