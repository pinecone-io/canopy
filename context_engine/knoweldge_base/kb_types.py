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


