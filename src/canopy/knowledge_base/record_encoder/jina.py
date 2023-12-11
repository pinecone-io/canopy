from pinecone_text.dense import JinaEncoder
from canopy.knowledge_base.record_encoder.dense import DenseRecordEncoder


class JinaRecordEncoder(DenseRecordEncoder):
    """
    JinaRecordEncoder is a type of DenseRecordEncoder that uses the JinaAI `embeddings` API.
    The implementation uses the `JinaEncoder` class from the `pinecone-text` library.
    For more information about see: https://github.com/pinecone-io/pinecone-text

    """  # noqa: E501

    def __init__(self,
                 *,
                 model_name: str = "jina-embeddings-v2-base-en",
                 batch_size: int = 400,
                 **kwargs):
        """
        Initialize the JinaRecordEncoder

        Args:
            model_name: The name of the embedding model to use.
            batch_size: The number of documents or queries to encode at once.
                        Defaults to 400.
            **kwargs: Additional arguments to pass to the underlying `pinecone-text. JinaEncoder`.
        """  # noqa: E501
        encoder = JinaEncoder(model_name, **kwargs)
        super().__init__(dense_encoder=encoder, batch_size=batch_size)
