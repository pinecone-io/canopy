from .reranker import TransparentReranker, Reranker

RERANKER_CLASSES = {
    cls.__name__: cls for cls in Reranker.__subclasses__()
}
