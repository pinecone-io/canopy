from typing import List

from .base import Chunker
from ..models import KBDocChunk
from canopy.tokenizer import Tokenizer
from ...models.data_models import Document


class TokenChunker(Chunker):
    """
    Simple chunker that splits a document into chunks (group of tokens) of a given size, using a tokenizer.
    A TokenChunker is a derived class of Chunker, which means that it can be referenced by a name
    and configured in a config file.
    """  # noqa: E501

    def __init__(self,
                 max_chunk_size: int = 256,
                 overlap: int = 30, ):
        """
        Using the global tokenizer, will set the class parameters for the TokenChunker.
        will check overlap and max_chunk_size.

        Args:
            max_chunk_size: size of the chunks, in tokens
            overlap: overlap between chunks, in tokens
        """  # noqa: E501

        # TODO: should add check for overlap not bigger than max_chunk_size
        if overlap < 0:
            cls_name = self.__class__.__name__
            raise ValueError(
                f"overlap for {cls_name} can't be negative, got: {overlap}"
            )

        if max_chunk_size <= 0:
            cls_name = self.__class__.__name__
            raise ValueError(
                f"max_chunk_size for {cls_name} must be positive, got: {max_chunk_size}"
            )

        self._tokenizer = Tokenizer()
        self._chunk_size = max_chunk_size
        self._overlap = overlap

    def chunk_single_document(self, document: Document) -> List[KBDocChunk]:
        """
        This methods takes a document and returns a list of KBDocChunks, where text is splitted
        evenly using the tokenizer. Firts the text is tokenized, then the tokens are splitted into chunks
        of a given size, with overlap between chunks.
        Last chunk is handled such that if the last chunk is smaller than the overlap, it will be removed.

        Args:
            document: document to be chunked

        Returns:
            text_chunks: list of chunks KBDocChunks from the document
        """  # noqa: E501
        tokens = self._tokenizer.tokenize(document.text)
        token_chunks = [tokens[i:i + self._chunk_size]
                        for i in range(0, len(tokens),
                                       self._chunk_size - self._overlap)]

        if len(token_chunks) == 0:
            return []

        # remove last chunk if it is smaller than overlap
        if len(token_chunks[-1]) <= self._overlap and len(token_chunks) > 1:
            token_chunks = token_chunks[:-1]

        text_chunks = [self._tokenizer.detokenize(chunk)
                       for chunk in token_chunks]
        return [KBDocChunk(id=f"{document.id}_{i}",
                           document_id=document.id,
                           text=text_chunk,
                           source=document.source,
                           metadata=document.metadata)
                for i, text_chunk in enumerate(text_chunks)]

    async def achunk_single_document(self, document: Document) -> List[KBDocChunk]:
        raise NotImplementedError()
