from typing import List

from .langchain_text_splitter import Language, RecursiveCharacterTextSplitter
from .recursive_character import RecursiveCharacterChunker
from context_engine.knoweldge_base.models import KBDocChunk
from context_engine.models.data_models import Document


class MarkdownChunker(RecursiveCharacterChunker):

    def __init__(self,
                 chunk_size: int = 256,
                 chunk_overlap: int = 0,
                 keep_separator: bool = True
                 ):
        separators = RecursiveCharacterTextSplitter.get_separators_for_language(
            Language.MARKDOWN
        )
        super().__init__(chunk_size=chunk_size,
                         chunk_overlap=chunk_overlap,
                         separators=separators,
                         keep_separator=keep_separator)

    async def achunk_single_document(self,
                                     document: Document) -> List[KBDocChunk]:
        raise NotImplementedError()
