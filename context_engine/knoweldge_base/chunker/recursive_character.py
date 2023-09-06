from copy import deepcopy
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

from context_engine.knoweldge_base.chunker.base import Chunker
from context_engine.knoweldge_base.models import KBDocChunk
from context_engine.knoweldge_base.tokenizer.tokenizer import Tokenizer
from context_engine.models.data_models import Document


class RecursiveCharacterChunker(Chunker):

    def __init__(self,
                 chunk_size: int = 256,
                 chunk_overlap: int = 0,
                 separators: Optional[List[str]] = None,
                 keep_separator: bool = True,
                 ):
        self._tokenizer = Tokenizer()
        self._chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._tokenizer.token_count,
            separators=separators,
            keep_separator=keep_separator)

    def chunk_single_document(self, document: Document) -> List[KBDocChunk]:
        text_chunks = self._chunker.split_text(document.text)
        return [KBDocChunk(id=f"{document.id}_{i}",
                           document_id=document.id,
                           text=text_chunk,
                           metadata=deepcopy(document.metadata))
                for i, text_chunk in enumerate(text_chunks)]

    async def achunk_single_document(self, document: Document) -> List[KBDocChunk]:
        raise NotImplementedError()
