from typing import List
from canopy.tokenizer.base import BaseTokenizer
from canopy.models.data_models import Messages


class StubTokenizer(BaseTokenizer):

    def __init__(self, message_overhead: int = 3):
        self._message_overhead = message_overhead

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def detokenize(self, tokens: List[str]) -> str:
        if not isinstance(tokens, List):
            raise TypeError(f"detokenize expect List[str], got f{type(tokens)}")
        return " ".join(tokens)

    def messages_token_count(self, messages: Messages) -> int:
        return sum(len(self.tokenize(msg.content)) + self._message_overhead
                   for msg in messages)
