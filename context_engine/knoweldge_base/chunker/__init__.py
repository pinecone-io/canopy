from .token_chunker import TokenChunker
from .markdown import MarkdownChunker
from .base import Chunker

CHUNKER_CLASSES = {
    cls.__name__: cls for cls in Chunker.__subclasses__()
}