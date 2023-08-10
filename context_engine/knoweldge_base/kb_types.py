from enum import Enum

from context_engine.knoweldge_base.rerankers.reranker import TransparentReranker
from context_engine.knoweldge_base.tokenizers.openai_tokenizer import OpenAITokenizer

TOKENIZER_TYPES = {
    "OpenAI": OpenAITokenizer,
}


CHUNKER_TYPES = {
    "markdown": NotImplemented,
    "html": NotImplemented,
    "character": NotImplemented,
}

RERANKER_TYPES = {
    "no_reranking": TransparentReranker,
}


def type_from_str(type_str: str, type_dict: dict, name: str) -> type:
    if type_str not in type_dict:
        raise ValueError(f"Unknown {name}: {type_str}. Allowed values are {list(type_dict.keys())}")
    return type_dict[type_str]