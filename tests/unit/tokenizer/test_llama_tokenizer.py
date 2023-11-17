import pytest
from canopy.tokenizer import LlamaTokenizer
from canopy.models.data_models import MessageBase, Role
from .base_test_tokenizer import BaseTestTokenizer


class TestLlamaTokenizer(BaseTestTokenizer):

    @staticmethod
    @pytest.fixture(scope="class")
    def tokenizer():
        return LlamaTokenizer(model_name="openlm-research/open_llama_7b_v2")

    @staticmethod
    @pytest.fixture
    def expected_tokens(text):
        return ['<s>', 'string', 'with', 'special', 'characters', 'like', '!', '@', 
                '#', '$', '%', '^', '&', '*', '()', '_+', '', '日', '本', 'spaces',
                '  ', '\n', '', '\n', '\n', 'C', 'ASE', 'c', 'A', 'se','']

    @staticmethod
    def test_token_count_empty_string(tokenizer):
        assert tokenizer.token_count("") == 1
        
    @staticmethod
    def test_tokenize_empty_string(tokenizer):
        assert tokenizer.tokenize("") == ['<s>']
        
    @staticmethod
    def test_messages_token_count(tokenizer):
        messages = [MessageBase(role=Role.USER, content="Hello, assistant.")]
        assert tokenizer.messages_token_count(messages) == 13

        messages = [MessageBase(role=Role.USER,
                                content="Hello, assistant."),
                    MessageBase(role=Role.ASSISTANT,
                                content="Hello, user. How can I assist you?")]
        assert tokenizer.messages_token_count(messages) == 29
        
    @staticmethod
    def test_messages_token_count_empty_messages(tokenizer):
        assert tokenizer.messages_token_count([]) == 3

    @staticmethod
    def test_special_tokens_to_natural_text(tokenizer):
        tokens = tokenizer.tokenize("<|endoftext|>")           
        assert tokens == ['<s>', '<', '|', 'end', 'of', 'text', '|', '>']
        
        assert tokenizer.detokenize(tokens) == "<s><|endoftext|>"
        
        assert tokenizer.token_count("<|endoftext|>") == 8
