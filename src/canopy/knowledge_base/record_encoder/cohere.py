from typing import List
from pinecone_text.dense.cohere_encoder import CohereEncoder
from canopy.knowledge_base.models import KBDocChunk, KBEncodedDocChunk, KBQuery
from canopy.knowledge_base.record_encoder.dense import DenseRecordEncoder
from canopy.models.data_models import Query


class CohereRecordEncoder(DenseRecordEncoder):
    """
    CohereRecordEncoder is a type of DenseRecordEncoder that uses the Cohere `embed` API.
    The implementation uses the `CohereEncoder` class from the `pinecone-text` library.
    For more information about see: https://github.com/pinecone-io/pinecone-text

    """  # noqa: E501

    def __init__(
        self,
        *,
        model_name: str = "embed-english-v3.0",
        batch_size: int = 100,
        **kwargs,
    ):
        """
        Initialize the CohereRecordEncoder

        Args:
            model_name: The name of the Cohere embeddings model to use for encoding. See https://docs.cohere.com/reference/embed
            batch_size: The number of documents or queries to encode at once.
                        Defaults to 400.
            **kwargs: Additional arguments to pass to the underlying `pinecone-text. CohereEncoder`.
        """  # noqa: E501
        encoder = CohereEncoder(model_name, **kwargs)
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

    async def _aencode_documents_batch(
        self, documents: List[KBDocChunk]
    ) -> List[KBEncodedDocChunk]:
        raise NotImplementedError

    async def _aencode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        raise NotImplementedError
