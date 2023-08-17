from typing import List

from .base import Chunker
from ..models import KBDocChunk
from ..tokenizer.base import Tokenizer
from ...models.data_models import Document


class TokenChunker(Chunker):

    def __init__(self,
                 tokenizer: Tokenizer,
                 max_chunk_size: int = 200,
                 overlap: int = 0, ):
        self._tokenizer = tokenizer
        self._chunk_size = max_chunk_size
        self._overlap = overlap

    def chunk_single_document(self, document: Document) -> List[KBDocChunk]:
        tokens = self._tokenizer.tokenize(document.text)
        token_chunks = [tokens[i:i + self._chunk_size]
                        for i in range(0, len(tokens), self._chunk_size - self._overlap)]

        # remove last chunk if it is smaller than overlap
        if len(token_chunks[-1]) <= self._overlap and len(token_chunks) > 1:
            token_chunks = token_chunks[:-1]

        text_chunks = [self._tokenizer.detokenize(chunk).strip()
                       for chunk in token_chunks]
        return [KBDocChunk(id=f"{document.id}_{i}",
                           document_id=document.id,
                           text=text_chunk,
                           metadata=document.metadata)
                for i, text_chunk in enumerate(text_chunks)]

    async def achunk_single_document(self, document: Document) -> List[KBDocChunk]:
        raise NotImplementedError()
