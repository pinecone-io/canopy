import tiktoken
from typing import List
from .base import Tokenizer


class OpenAITokenizer(Tokenizer):
    def __init__(self, model_name: str):
        self._encoder = tiktoken.encoding_for_model(model_name)

    def tokenize(self, text: str) -> List[str]:
        return [self._encoder.decode([encoded_token])
                for encoded_token in self._encoder.encode(text)]

    def detokenize(self, tokens: List[str]) -> str:
        if not isinstance(tokens, List):
            raise TypeError(f"detokenize expect List[str], got f{type(tokens)}")
        return "".join(tokens)

    def token_count(self, text: str) -> int:
        return len(self._encoder.encode(text))
