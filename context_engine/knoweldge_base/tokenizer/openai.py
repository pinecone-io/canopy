from typing import List

import tiktoken

from .base import Tokenizer


class OpenAITokenizer(Tokenizer):
    def __init__(self, model_name: str):
        self._encoder = tiktoken.encoding_for_model(model_name)

    def encode(self, text: str) -> List[int]:
        return self._encoder.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self._encoder.decode(tokens)

    def token_count(self, text: str) -> int:
        return len(self.encode(text))
