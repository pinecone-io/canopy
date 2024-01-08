from canopy.knowledge_base.record_encoder import OpenAIRecordEncoder


class AzureOpenAIRecordEncoder(OpenAIRecordEncoder):
    """
    AzureOpenAIRecordEncoder is a type of DenseRecordEncoder that uses the OpenAI `embeddings` API.
    The implementation uses the `AzureOpenAIEncoder` class from the `pinecone-text` library.
    For more information about see: https://github.com/pinecone-io/pinecone-text

    """  # noqa: E501

    def __init__(
        self,
        *,
        model_name: str = "text-embedding-ada-002",
        batch_size: int = 400,
        **kwargs
    ):
        """
        Initialize the AzureOpenAIRecordEncoder

        Args:
            model_name: The name of the OpenAI embeddings model to use for encoding. See https://platform.openai.com/docs/models/embeddings
            batch_size: The number of documents or queries to encode at once.
                        Defaults to 400.
            **kwargs: Additional arguments to pass to the underlying `pinecone-text. OpenAIEncoder`.
        """  # noqa: E501
        encoder = AzureOpenAIEncoder(model_name, **kwargs)
        super().__init__(dense_encoder=encoder, batch_size=batch_size)
