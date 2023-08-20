from ..stubs.stub_tokenizer import StubTokenizer


class TestBaseTokenizer:

    @classmethod
    def setup_class(cls):
        cls.tokenizer = StubTokenizer()
        cls.text = "1 2 3 4 5"

    def test_token_count(self):
        assert self.tokenizer.token_count(self.text) == 5
        assert self.tokenizer.token_count("") == 0
