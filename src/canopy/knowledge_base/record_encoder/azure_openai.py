import os

from pinecone_text.dense import AzureOpenAIEncoder

from canopy.knowledge_base.record_encoder import OpenAIRecordEncoder, DenseRecordEncoder
import openai


class AzureOpenAIRecordEncoder(OpenAIRecordEncoder):
    """
    AzureOpenAIRecordEncoder is a type of DenseRecordEncoder that uses the Azure OpenAI's `embeddings` deployments.
    The implementation uses the `AzureOpenAIEncoder` class from the `pinecone-text` library.
    For more information about see: https://github.com/pinecone-io/pinecone-text

    Azure OpenAI services require a valid API key, and an Azure endpoint. You will need
    To set the following environment variables:
    - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key.
    - AZURE_OPENAI_ENDPOINT: Your Azure endpoint, including the resource, e.g. `https://example-resource.azure.openai.com/`
    """  # noqa: E501

    def __init__(
            self,
            *,
            model_name: str,
            api_version: str = "2023-12-01-preview",
            batch_size: int = 400,
            **kwargs
    ):
        """
        Initialize the AzureOpenAIRecordEncoder

        Args:
            model_name: The name of embeddings model deployment to use for encoding
            api_version: The Azure OpenAI API version to use. Defaults to "2023-12-01-preview".
            batch_size: The number of documents or queries to encode at once.
                        Defaults to 400.
            **kwargs: Additional arguments to pass to the underlying `pinecone-text.AzureOpenAIEncoder`.
        """  # noqa: E501
        try:
            encoder = AzureOpenAIEncoder(model_name, api_version=api_version, **kwargs)
        except (openai.OpenAIError, ValueError) as e:
            raise RuntimeError(
                "Failed to connect to Azure OpenAI, please make sure that the "
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables "
                "are set correctly. "
                f"Underlying Error:\n{self._format_openai_error(e)}"
            ) from e

        DenseRecordEncoder.__init__(self, dense_encoder=encoder, batch_size=batch_size,
                                    **kwargs)

    def _format_error(self, err):
        if isinstance(err, openai.AuthenticationError):
            return (
                "Failed to connect to Azure OpenAI, please make sure that the "
                "AZURE_OPENAI_API_KEY environment variable is set correctly. "
                f"Underlying Error:\n{self._format_openai_error(err)}"
            )
        elif isinstance(err, openai.APIConnectionError):
            return (
                f"Failed to connect to your Azure OpenAI endpoint, please make sure "
                f"that the provided endpoint {os.getenv('AZURE_OPENAI_ENDPOINT')} "
                f"is correct. Underlying Error:\n{self._format_openai_error(err)}"
            )
        elif isinstance(err, openai.NotFoundError):
            return (
                f"Failed to connect to your Azure OpenAI. Please make sure that "
                f"you have provided the correct deployment name: {self.model_name} "
                f"and API version: {self._client._api_version}. "
                f"Underlying Error:\n{self._format_openai_error(err)}"
            )
        else:
            return super()._format_error(err)
