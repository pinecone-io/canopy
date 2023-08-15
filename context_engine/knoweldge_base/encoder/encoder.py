from typing import List

from context_engine.knoweldge_base.encoder.steps.base import Encoder
from context_engine.knoweldge_base.models import KBDocChunk, KBQuery, KBEncodedDocChunk
from context_engine.models.data_models import Query


class EncodingPipeline:

    def __init__(self, pipeline: List[Encoder], batch_size: int = 1):
        if len(enconding_steps) == 0:
            raise ValueError("Must provide at least one encoding step")

        if enconding_steps[0].__class__.__name__ != "DenseEncodingStep":
            raise ValueError("First encoding step must be a DenseEncodingStep")

        self.encoding_steps = enconding_steps
        self.batch_size = batch_size

    @staticmethod
    def _batch_iterator(data: list, batch_size):
        return (data[pos:pos + batch_size] for pos in range(0, len(data), batch_size))

    def encode_documents(self, records: List[PineconeDocumentRecord]
                         ) -> List[PineconeDocumentRecord]:
        for batch in self._batch_iterator(records, self.batch_size):
            for step in self.encoding_steps:
                # TODO: understand if this is the best way to do this, or should it return a new list
                # Each step is editing the encoded_chunks in place
                step.encode_documents(batch)
        return records

    def encode_queries(self, records: List[PineconeQueryRecord]
                       ) -> List[PineconeQueryRecord]:
        for batch in self._batch_iterator(records, self.batch_size):
            for step in self.encoding_steps:
                # Each step is editing the kb_queries in place
                step.encode_queries(batch)
        return records

    async def aencode_documents(self, documents: List[KBDocChunk]
                                ) -> List[KBEncodedDocChunk]:
        pass

    async def aencode_queries(self, queries: List[Query]
                              ) -> List[KBQuery]:
        pass


