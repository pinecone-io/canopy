from ...utils import TypeDict as _TypeDict

from .token_chunker import TokenChunker
from .markdown import MarkdownChunker
from .base import Chunker

CHUNKER_CLASSES: _TypeDict = {
    cls.__name__: cls for cls in Chunker.__subclasses__()
}
CHUNKER_CLASSES['default'] = MarkdownChunker
