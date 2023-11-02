from copy import deepcopy
from typing import List, Optional

from .langchain_text_splitter import RecursiveCharacterTextSplitter

from canopy.knowledge_base.chunker.base import Chunker
from canopy.knowledge_base.models import KBDocChunk
from canopy.tokenizer import Tokenizer
from canopy.models.data_models import Document


class RecursiveCharacterChunker(Chunker):
    """
    A chunker that splits a document into chunks of a given size, using a recursive character splitter.
    A RecursiveCharacterChunker is a derived class of Chunker, which means that it can be referenced by a name
    and configured in a config file.
    """  # noqa: E501

    def __init__(self,
                 chunk_size: int = 256,
                 chunk_overlap: int = 0,
                 separators: Optional[List[str]] = None,
                 keep_separator: bool = True,
                 ):
        """
        RecursiveCharacterTextSplitter is a text splitter from the langchain library.
        It splits a text into chunks of a given size, using a recursive character splitter.

        Args:
            chunk_size: size of the chunks, in tokens
            chunk_overlap: overlap between chunks
            separators: list of separators to use for splitting the text
            keep_separator: whether to keep the separator in the chunk or not
        """  # noqa: E501
        self._tokenizer = Tokenizer()
        self._chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._tokenizer.token_count,
            separators=separators,
            keep_separator=keep_separator)

    def chunk_single_document(self, document: Document) -> List[KBDocChunk]:
        """
        using the RecursiveCharacterTextSplitter, this method takes a document and returns a list of KBDocChunks
        Args:
            document: document to be chunked

        Returns:
            chunks: list of chunks KBDocChunks from the document, where text is splitted
                              evenly using the RecursiveCharacterTextSplitter
        """  # noqa: E501
        # TODO: check overlap not bigger than max_chunk_size
        text_chunks = self._chunker.split_text(document.text)
        return [KBDocChunk(id=f"{document.id}_{i}",
                           document_id=document.id,
                           text=text_chunk,
                           source=document.source,
                           metadata=deepcopy(document.metadata))
                for i, text_chunk in enumerate(text_chunks)]

    async def achunk_single_document(self, document: Document) -> List[KBDocChunk]:
        raise NotImplementedError()
