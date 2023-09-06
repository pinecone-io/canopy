from typing import List
from context_engine.knoweldge_base.tokenizer.tokenizer import BaseTokenizer
from context_engine.models.data_models import Messages


class StubTokenizer(BaseTokenizer):

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def detokenize(self, tokens: List[str]) -> str:
        if not isinstance(tokens, List):
            raise TypeError(f"detokenize expect List[str], got f{type(tokens)}")
        return " ".join(tokens)

    def messages_token_count(self, messages: Messages) -> int:
        return sum(len(self.tokenize(msg.content)) + 3 for msg in messages)
