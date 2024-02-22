from typing import Optional
from pinecone_text.dense import SentenceTransformerEncoder
from canopy.knowledge_base.record_encoder.dense import DenseRecordEncoder
from huggingface_hub.utils import RepositoryNotFoundError


class SentenceTransformerRecordEncoder(DenseRecordEncoder):
    """
    SentenceTransformerRecordEncoder is a type of DenseRecordEncoder that uses a Sentence Transformer model.
    The implementation uses the `SentenceTransformerEncoder` class from the `pinecone-text` library.
    For more information about see: https://github.com/pinecone-io/pinecone-text

    """  # noqa: E501

    def __init__(self,
                 *,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 query_encoder_name: Optional[str] = None,
                 batch_size: int = 400,
                 device: Optional[str] = None,
                 **kwargs) -> None:
        """
        Initialize the SentenceTransformerRecordEncoder

        Args:
            model_name: The name of the embedding model to use for encoding documents.
                        See https://huggingface.co/models?library=sentence-transformers
                        for all possible Sentence Transformer models.
            query_encoder_name: The name of the embedding model to use for encoding queries.
                        See https://huggingface.co/models?library=sentence-transformers
                        for all possible Sentence Transformer models.
                        Defaults to `model_name`.
            batch_size: The number of documents or queries to encode at once.
                        Defaults to 400.
            device: The local device to use for encoding, for example "cpu", "cuda" or "mps".
                        Defaults to "cuda" if cuda is available, otherwise to "cpu".
            **kwargs: Additional arguments to pass to the underlying `pinecone-text.SentenceTransformerEncoder`.
        """  # noqa: E501
        try:
            encoder = SentenceTransformerEncoder(
                document_encoder_name=model_name,
                query_encoder_name=query_encoder_name,
                device=device,
                **kwargs,
            )
        except RepositoryNotFoundError as e:
            raise RuntimeError(
                "Your chosen Sentence Transformer model(s) could not be found. "
                f"Details: {str(e)}"
            ) from e
        except ImportError:
            raise ImportError(
                f"{self.__class__.__name__} requires the `torch` and `transformers` "
                f"extra dependencies. Please install them using "
                f"`pip install canopy-sdk[torch,transformers]`."
            )
        super().__init__(dense_encoder=encoder, batch_size=batch_size)
