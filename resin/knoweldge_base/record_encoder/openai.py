from typing import List
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from pinecone_text.dense.openai_encoder import OpenAIEncoder
from resin.knoweldge_base.models import KBDocChunk, KBEncodedDocChunk, KBQuery
from resin.knoweldge_base.record_encoder.dense import DenseRecordEncoder
from resin.models.data_models import Query
from resin.utils.openai_exceptions import OPEN_AI_TRANSIENT_EXCEPTIONS


class OpenAIRecordEncoder(DenseRecordEncoder):

    def __init__(self,
                 *,
                 model_name: str = "text-embedding-ada-002",
                 batch_size: int = 100,
                 **kwargs):
        encoder = OpenAIEncoder(model_name)
        super().__init__(dense_encoder=encoder, batch_size=batch_size, **kwargs)

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(OPEN_AI_TRANSIENT_EXCEPTIONS),
    )
    def encode_documents(self, documents: List[KBDocChunk]) -> List[KBEncodedDocChunk]:
        return super().encode_documents(documents)

    async def _aencode_documents_batch(self,
                                       documents: List[KBDocChunk]
                                       ) -> List[KBEncodedDocChunk]:
        raise NotImplementedError

    async def _aencode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        raise NotImplementedError
