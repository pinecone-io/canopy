from typing import List

from openai import APIError, RateLimitError
from pinecone_text.dense.openai_encoder import OpenAIEncoder
from canopy.knowledge_base.models import KBDocChunk, KBEncodedDocChunk, KBQuery
from canopy.knowledge_base.record_encoder.dense import DenseRecordEncoder
from canopy.models.data_models import Query

OPENAI_AUTH_ERROR_MSG = (
    "Failed to connect to OpenAI, please make sure that the OPENAI_API_KEY "
    "environment variable is set correctly.\n"
)


def _format_openai_error(e):
    try:
        return e.response.json()['error']['message']
    except:
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
        except APIError as e:
            raise RuntimeError(
                OPENAI_AUTH_ERROR_MSG + f"Error: {_format_openai_error(e)}"
            )
        super().__init__(dense_encoder=encoder, batch_size=batch_size)

    def _encode_documents_batch(self,
                                documents: List[KBDocChunk]
                                ) -> List[KBEncodedDocChunk]:
        try:
            return super()._encode_documents_batch(documents)
        except RateLimitError as e:
            raise RuntimeError(
                f"Your OpenAI account seem to have reached the rate limit. "
                f"Error: {_format_openai_error(e)}"
            )
        except APIError as e:
            raise RuntimeError(
                f"Failed to encode documents using OpenAI embeddings model. "
                f"Error: {_format_openai_error(e)}"
            )

    def _encode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        try:
            return super()._encode_queries_batch(queries)
        except RateLimitError as e:
            raise RuntimeError(
                f"Your OpenAI account seem to have reached the rate limit. "
                f"Error: {_format_openai_error(e)}"
            )
        except APIError as e:
            raise RuntimeError(
                f"Failed to encode queries using OpenAI embeddings model. "
                f"Error: {_format_openai_error(e)}"
            )

    async def _aencode_documents_batch(self,
                                       documents: List[KBDocChunk]
                                       ) -> List[KBEncodedDocChunk]:
        raise NotImplementedError

    async def _aencode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        raise NotImplementedError
