import os
from typing import List
from pinecone_text.dense.openai_encoder import OpenAIEncoder
from canopy.knowledge_base.models import KBDocChunk, KBEncodedDocChunk, KBQuery
from canopy.knowledge_base.record_encoder.dense import DenseRecordEncoder
from canopy.models.data_models import Query

ANYSCALE_BASE_URL = "https://api.endpoints.anyscale.com/v1"


class AnyscaleRecordEncoder(DenseRecordEncoder):
    """
    AnyscaleRecordEncoder is a type of DenseRecordEncoder that uses the OpenAI `embeddings` API.
    The implementation uses the `OpenAIEncoder` class from the `pinecone-text` library.
    For more information about see: https://github.com/pinecone-io/pinecone-text

    """  # noqa: E501
    """
    Initialize the AnyscaleRecordEncoder

    Args:
        api_key: The Anyscale Endpoint API Key
        base_url: The Base URL for Anyscale Endpoint
        model_name: The name of the Anyscale embeddings model to use for encoding. See https://docs.endpoints.anyscale.com/category/supported-models
        batch_size: The number of documents or queries to encode at once.
                    Defaults to 400.
        **kwargs: Additional arguments to pass to the underlying `pinecone-text. OpenAIEncoder`.
    """  # noqa: E501
    def __init__(self,
                 *,
                 api_key: str = "",
                 base_url: str = ANYSCALE_BASE_URL,
                 model_name: str = "thenlper/gte-large",
                 batch_size: int = 400,
                 **kwargs):

        ae_api_key = api_key or os.environ.get("ANYSCALE_API_KEY")
        if not ae_api_key:
            raise ValueError(
                "Anyscale API key is required to use Anyscale. "
                "Please provide it as an argument "
                "or set the ANYSCALE_API_KEY environment variable."
            )
        ae_base_url = base_url
        encoder = OpenAIEncoder(model_name,
                                base_url=ae_base_url, api_key=ae_api_key,
                                **kwargs)
        super().__init__(dense_encoder=encoder, batch_size=batch_size)

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
