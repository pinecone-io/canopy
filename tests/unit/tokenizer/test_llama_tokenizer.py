import pytest
from canopy.tokenizer import LlamaTokenizer
from canopy.models.data_models import MessageBase, Role
from .base_test_tokenizer import BaseTestTokenizer


class TestLlamaTokenizer(BaseTestTokenizer):
    @staticmethod
    @pytest.fixture(scope="class")
    def tokenizer():
        return LlamaTokenizer(model_name="hf-internal-testing/llama-tokenizer")

    @staticmethod
    @pytest.fixture
    def expected_tokens(text):
        return [
            "▁string",
            "▁with",
            "▁special",
            "▁characters",
            "▁like",
            "▁!",
            "@",
            "#",
            "$",
            "%",
            "^",
            "&",
            "*",
            "()",
            "_+",
            "▁",
            "日",
            "本",
            "▁spaces",
            "▁▁▁",
            "<0x0A>",
            "▁",
            "<0x0A>",
            "<0x0A>",
            "▁CASE",
            "▁c",
            "A",
            "se",
            "▁",
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
    def test_messages_token_count_empty_messages(tokenizer):
        assert tokenizer.messages_token_count([]) == 3

    @staticmethod
    def test_special_tokens_to_natural_text(tokenizer):
        input_text = "</s>_<0x0A>__ <unk><s>word"
        tokens = tokenizer.tokenize(input_text)
        expected_tokens = [
            "▁</s>",
            "_",
            "<",
            "0",
            "x",
            "0",
            "A",
            ">",
            "__",
            "▁<unk>",
            "<",
            "s",
            ">",
            "word",
        ]
        assert tokens == expected_tokens

        # TODO: this currently fails since detokenize() adds a space after <s> and </s>.
        #  We need to decide if this is the desired behavior or not.
        assert tokenizer.detokenize(tokens) == input_text

        assert tokenizer.token_count(input_text) == len(tokens)
