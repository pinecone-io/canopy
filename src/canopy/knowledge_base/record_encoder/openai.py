from typing import List

from openai import OpenAIError, RateLimitError, APIConnectionError, AuthenticationError
from pinecone_text.dense.openai_encoder import OpenAIEncoder
from canopy.knowledge_base.models import KBDocChunk, KBEncodedDocChunk, KBQuery
from canopy.knowledge_base.record_encoder.dense import DenseRecordEncoder
from canopy.models.data_models import Query


def _format_openai_error(e):
    try:
        return e.response.json()['error']['message']
    except Exception:
        return str(e)


class OpenAIRecordEncoder(DenseRecordEncoder):
    """
    OpenAIRecordEncoder is a type of DenseRecordEncoder that uses the OpenAI `embeddings` API.
    The implementation uses the `OpenAIEncoder` class from the `pinecone-text` library.
    For more information about see: https://github.com/pinecone-io/pinecone-text

    """  # noqa: E501

    def __init__(self,
                 *,
                 model_name: str = "text-embedding-ada-002",
                 batch_size: int = 400,
                 **kwargs):
        """
        Initialize the OpenAIRecordEncoder

        Args:
            model_name: The name of the OpenAI embeddings model to use for encoding. See https://platform.openai.com/docs/models/embeddings
            batch_size: The number of documents or queries to encode at once.
                        Defaults to 400.
            **kwargs: Additional arguments to pass to the underlying `pinecone-text. OpenAIEncoder`.
        """  # noqa: E501
        try:
            encoder = OpenAIEncoder(model_name, **kwargs)
        except OpenAIError as e:
            raise RuntimeError(
                "Failed to connect to OpenAI, please make sure that the OPENAI_API_KEY "
                "environment variable is set correctly.\n"
                f"Error: {_format_openai_error(e)}"
            ) from e
        super().__init__(dense_encoder=encoder, batch_size=batch_size)

    async def _aencode_documents_batch(self,
                                       documents: List[KBDocChunk]
                                       ) -> List[KBEncodedDocChunk]:
        raise NotImplementedError

    async def _aencode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        raise NotImplementedError

    def _format_error(self, err):
        if isinstance(err, RateLimitError):
            return (f"Your OpenAI account seem to have reached the rate limit. "
                    f"Details: {_format_openai_error(err)}")
        elif isinstance(err, (AuthenticationError, APIConnectionError)):
            return (f"Failed to connect to OpenAI, please make sure that the "
                    f"OPENAI_API_KEY environment variable is set correctly. "
                    f"Details: {_format_openai_error(err)}")
        else:
            return _format_openai_error(err)
