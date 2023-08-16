from context_engine.knoweldge_base.tokenizer import OpenAITokenizer


class TestOpenAITokenizer:

    @classmethod
    def setup_class(cls):
        cls.tokenizer = OpenAITokenizer(model_name="gpt-3.5-turbo")
        cls.text = "string with special characters like !@#$%^&*()_+ 日本 spaces   \n \n\n CASE cAse "
        cls.expected_tokens = ['string', ' with', ' special', ' characters', ' like',
                               ' !', '@', '#$', '%^', '&', '*', '()', '_', '+', ' 日',
                               '本', ' spaces', '   \n', ' \n\n', ' CASE', ' c', 'A', 'se', " "]

    def test_tokenize(self):
        tokens = self.tokenizer.tokenize(self.text)
        assert tokens == self.expected_tokens

        assert self.tokenizer.tokenize("") == []

    def test_detokenize(self):
        text = self.tokenizer.detokenize(self.expected_tokens)
        assert text == self.text

        assert self.tokenizer.detokenize([]) == ""

    def test_token_count(self):
        token_count = self.tokenizer.token_count(self.text)
        assert token_count == len(self.expected_tokens)

        assert self.tokenizer.token_count("") == 0

    def test_tokenize_detokenize_compatibility(self):
        assert self.tokenizer.detokenize(self.tokenizer.tokenize(self.text)) == self.text
        assert self.tokenizer.tokenize(self.tokenizer.detokenize(self.expected_tokens)) == self.expected_tokens
