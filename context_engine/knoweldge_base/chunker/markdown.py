from typing import List

from context_engine.knoweldge_base.chunker.recursive_character \
    import RecursiveCharacterChunker
from context_engine.knoweldge_base.models import KBDocChunk
from context_engine.models.data_models import Document


class MarkdownChunker(RecursiveCharacterChunker):

    # taken from langchain with small modifications
    MARKDOWN_SEPARATORS = [
                "\n# "
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n##### ",
                "\n###### ",
                # Note the alternative syntax for headings (below) is not handled here
                # Heading level 2
                # ---------------
                # End of code block
                "```\n\n",
                '|\n\n'
                # Horizontal lines
                "\n\n***\n\n",
                "\n\n---\n\n",
                "\n\n___\n\n",
                # Note that this splitter doesn't handle horizontal lines defined
                # by *three or more* of ***, ---, or ___, but this is not handled
                "\n\n",
                "\n",
                " ",
                "",
            ]

    def __init__(self,
                 chunk_size: int = 256,
                 chunk_overlap: int = 0,
                 keep_separator: bool = True
                 ):
        super().__init__(chunk_size=chunk_size,
                         chunk_overlap=chunk_overlap,
                         separators=MarkdownChunker.MARKDOWN_SEPARATORS,
                         keep_separator=keep_separator)

    async def achunk_single_document(self,
                                     document: Document) -> List[KBDocChunk]:
        raise NotImplementedError()
