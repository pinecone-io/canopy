import tiktoken
from typing import List
from .base import BaseTokenizer
from ..models.data_models import Messages


class OpenAITokenizer(BaseTokenizer):

    MESSAGE_TOKENS_OVERHEAD = 3
    FIXED_PREFIX_TOKENS = 3

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self._encoder = tiktoken.encoding_for_model(model_name)

    def tokenize(self, text: str) -> List[str]:
        return [self._encoder.decode([encoded_token])
                for encoded_token in self._encode(text)]

    def detokenize(self, tokens: List[str]) -> str:
        if not isinstance(tokens, List):
            raise TypeError(f"detokenize expect List[str], got f{type(tokens)}")
        return "".join(tokens)

    def token_count(self, text: str) -> int:
        return len(self._encode(text))

    def _encode(self, text):
        return self._encoder.encode(text, disallowed_special=())

    def messages_token_count(self, messages: Messages) -> int:
        # Adapted from: https://github.com/openai/openai-cookbook/.../How_to_format_inputs_to_ChatGPT_models.ipynb # noqa
        num_tokens = 0
        for message in messages:
            num_tokens += self.MESSAGE_TOKENS_OVERHEAD
            for key, value in message.dict().items():
                num_tokens += self.token_count(value)
        num_tokens += self.FIXED_PREFIX_TOKENS
        return num_tokens
