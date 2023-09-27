from typing import List

from pinecone_text.dense.openai_encoder import OpenAIEncoder
from resin.knoweldge_base.models import KBDocChunk, KBEncodedDocChunk, KBQuery
from resin.knoweldge_base.record_encoder.dense import DenseRecordEncoder
from resin.models.data_models import Query


class OpenAIRecordEncoder(DenseRecordEncoder):
    DEFAULT_MODEL_NAME = "text-embedding-ada-002"

    def __init__(self,
                 *,
                 model_name: str = DEFAULT_MODEL_NAME,
                 batch_size: int = 100,
                 **kwargs):
        encoder = OpenAIEncoder(model_name)
        super().__init__(dense_encoder=encoder, batch_size=batch_size, **kwargs)

    async def _aencode_documents_batch(self,
                                       documents: List[KBDocChunk]
                                       ) -> List[KBEncodedDocChunk]:
        raise NotImplementedError

    async def _aencode_queries_batch(self, queries: List[Query]) -> List[KBQuery]:
        raise NotImplementedError
