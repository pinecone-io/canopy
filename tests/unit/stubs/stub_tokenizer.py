from typing import List
from context_engine.knoweldge_base.tokenizer.base import Tokenizer


class StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def detokenize(self, tokens: List[str]) -> str:
        return " ".join(tokens)
