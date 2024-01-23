from pinecone_text.dense import SentenceTransformerEncoder
from canopy.knowledge_base.record_encoder.dense import DenseRecordEncoder


class SentenceTransformerRecordEncoder(DenseRecordEncoder):
    """
    SentenceTransformerRecordEncoder is a type of DenseRecordEncoder that uses a Sentence Transformer model.
    The implementation uses the `SentenceTransformerEncoder` class from the `pinecone-text` library.
    For more information about see: https://github.com/pinecone-io/pinecone-text

    """  # noqa: E501

    def __init__(self,
                 *,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 400,
                 **kwargs):
        """
        Initialize the SentenceTransformerRecordEncoder

        Args:
            model_name: The name of the embedding model to use. See https://huggingface.co/models?library=sentence-transformers
                        for all possible Sentence Transformer models.
            batch_size: The number of documents or queries to encode at once.
                        Defaults to 400.
            **kwargs: Additional arguments to pass to the underlying `pinecone-text.SentenceTransformerEncoder`.
        """  # noqa: E501
        encoder = SentenceTransformerEncoder(model_name, **kwargs)
        super().__init__(dense_encoder=encoder, batch_size=batch_size)
