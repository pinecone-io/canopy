from typing import List

from context_engine.knoweldge_base.encoder.steps.base import Encoder
from context_engine.knoweldge_base.models import KBQuery, KBEncodedDocChunk


class PipelineEncoder(Encoder):

    def __init__(self, encoding_steps: List[Encoder], **kwargs):
        super().__init__(**kwargs)
        if len(encoding_steps) == 0:
            raise ValueError("Must provide at least one encoding step")

        if encoding_steps[0].__class__.__name__ != "DenseEncodingStep":
            raise ValueError("First encoding step must be a DenseEncodingStep")

        self.encoding_steps = encoding_steps

    def _encode_documents_batch(self, documents: List[KBEncodedDocChunk]):
        for step in self.encoding_steps:
            # Each step is editing the encoded_chunks in place
            step.encode_documents(documents)

    def _encode_queries_batch(self, queries: List[KBQuery]):
        for step in self.encoding_steps:
            # Each step is editing the kb_queries in place
            step.encode_queries(queries)


