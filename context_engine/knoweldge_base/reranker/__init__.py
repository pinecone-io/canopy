from ...utils import TypeDict as _TypeDict

from .reranker import TransparentReranker, Reranker

RERANKER_CLASSES: _TypeDict = {
    cls.__name__: cls for cls in Reranker.__subclasses__()
}
RERANKER_CLASSES['default'] = TransparentReranker
