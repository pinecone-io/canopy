import pytest
from canopy.tokenizer import LlamaTokenizer
from .test_openai_tokenizer import TestOpenAITokenizer


class TestLlamaTokenizer(TestOpenAITokenizer):

    @staticmethod
    @pytest.fixture(scope="class")
    def tokenizer():
        return LlamaTokenizer(model_name="openlm-research/open_llama_7b_v2")

    @staticmethod
    @pytest.fixture
    def expected_tokens(text):
        return ['▁string', '▁with', '▁special', '▁characters', '▁like', '▁!', '@',
                '#', '$', '%', '^', '&', '*', '()', '_+', '▁', '日', '本', '▁spaces',
                '▁▁▁', '<0x0A>', '▁', '<0x0A>', '<0x0A>', '▁C', 'ASE', '▁c', 'A', 'se',
                '▁']

    @staticmethod
    def test_special_tokens_to_natural_text(tokenizer):
        input_text = "</s>_<0x0A>__ <unk><s>word"
        tokens = tokenizer.tokenize(input_text)
        expected_tokens = ['</s>', '▁_', '<', '0', 'x', '0', 'A', '>', '__', '▁',
                           '<unk>', '<s>', '▁word']
        assert tokens == expected_tokens
        
        # TODO: this currently fails since detokenize() adds a space after <s> and </s>.
        #  We need to decide if this is the desired behavior or not.
        assert tokenizer.detokenize(tokens) == input_text
        
        assert tokenizer.token_count(input_text) == len(tokens)
