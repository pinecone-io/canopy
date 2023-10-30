from typing import List

from .langchain_text_splitter import Language, RecursiveCharacterTextSplitter
from .recursive_character import RecursiveCharacterChunker
from canopy.knowledge_base.models import KBDocChunk
from canopy.models.data_models import Document


class MarkdownChunker(RecursiveCharacterChunker):
    """    
    MarkdownChunker is a subclass of RecursiveCharacterChunker that is configured 
    to chunk markdown documents. It uses RecursiveCharacterTextSplitter to split 
    the text of the document into chunks, by providing the separators for markdown documents 
    (also from LangChainTextSplitter, with modifications)

    Args:
        chunk_size (int, optional): size of the chunks. Defaults to 256 tokens.
        chunk_overlap (int, optional): overlap between chunks. Defaults to 0.
        keep_separator (bool, optional): whether to keep the separator in the chunk. Defaults to True.
    """ # noqa

    def __init__(self,
                 chunk_size: int = 256,
                 chunk_overlap: int = 0,
                 keep_separator: bool = True
                 ):
        """
        Iniitalizes RecursiveCharacterChunker with the separators for markdown documents.

        Args:
            chunk_size (int, optional): size of the chunks. Defaults to 256 tokens.
            chunk_overlap (int, optional): overlap between chunks. Defaults to 0.
            keep_separator (bool, optional): whether to keep the separator in the chunk. Defaults to True.

        """ # noqa
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
