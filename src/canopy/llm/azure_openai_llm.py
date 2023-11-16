from typing import Optional, Any

from openai import AzureOpenAI

from canopy.llm import OpenAILLM


AZURE_OPENAI_ENDPOINT = "https://devrel.openai.azure.com/"
AZURE_OPENAI_KEY = "d5e870aabbb14cb09a52038215c0fb37"


class AzureOpenAILLM(OpenAILLM):
    """

    Note: Azure OpenAI services requires a valid Azure API key and a valid Azure endpoint to use this class.
          You can set the "AZURE_OPENAI_KEY" and "AZURE_OPENAI_ENDPOINT" environment variables to your API key and
          endpoint, respectively, or you can directly, e.g.:
          >>> from openai import AzureOpenAI
          >>> AzureOpenAI.api_key = "YOUR_AZURE_API_KEY"
          >>> AzureOpenAI.api_base = "YOUR_AZURE_ENDPOINT"
    """
    def __init__(self,
                 api_type: str = "azure",
                 *,
                 openai_api_version: str,
                 model_name: str,  # not the same as openai model_name; this is azure deployment name
                 azure_api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs: Any
                 ):
        """
        """
        super().__init__(model_name)
        self.api_type = api_type
        self.openai_api_version = openai_api_version
        self.model_name = model_name
        self.azure_api_key = azure_api_key
        self.base_url = base_url

        self._client = AzureOpenAI(azure_endpoint=self.base_url,
                                   api_key=self.azure_api_key,
                                   api_version=self.openai_api_version
                                   )

        self.default_model_params = kwargs


if __name__ == "__main__":
    llm = AzureOpenAILLM(openai_api_version="2023-07-01-preview",
                         api_key=AZURE_OPENAI_KEY,
                         base_url=AZURE_OPENAI_ENDPOINT,
                         model_name="audrey_canopy_test")

    print('hi')