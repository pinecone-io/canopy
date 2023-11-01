from typing import List
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from pinecone_text.dense.openai_encoder import OpenAIEncoder
from canopy.knowledge_base.models import KBDocChunk, KBEncodedDocChunk, KBQuery
from canopy.knowledge_base.record_encoder.dense import DenseRecordEncoder
from canopy.models.data_models import Query
from canopy.utils.openai_exceptions import OPEN_AI_TRANSIENT_EXCEPTIONS


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
        encoder = OpenAIEncoder(model_name)
        super().__init__(dense_encoder=encoder, batch_size=batch_size, **kwargs)

    @retry(
        reraise=True,
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(OPEN_AI_TRANSIENT_EXCEPTIONS),
    )
    def encode_documents(self, documents: List[KBDocChunk]) -> List[KBEncodedDocChunk]:
        """
        Encode a list of documents, takes a list of KBDocChunk and returns a list of KBEncodedDocChunk.

        Args:
            documents: A list of KBDocChunk to encode.

        Returns:
            encoded chunks: A list of KBEncodedDocChunk, with the `values` field populated by the generated embeddings vector.
        """  # noqa: E501
        return super().encode_documents(documents)

    async def _aencode_documents_batch(self,
                                       documents: List[KBDocChunk]
                                       ) -> List[KBEncodedDocChunk]:
        raise NotImplementedError

    async def _aencode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        raise NotImplementedError
