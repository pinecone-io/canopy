from abc import ABC, abstractmethod
from typing import List, Union

from context_engine.knoweldge_base.encoder.base import Encoder
from context_engine.knoweldge_base.models import KBQuery, KBEncodedDocChunk

class ReEncode(ABC):
    @abstractmethod
    # Each step is editing the encoded_chunks in place
    def _encode_documents_batch(self, documents: List[KBEncodedDocChunk]):
        pass

    @abstractmethod
    # Each step is editing the kb_queries in place
    def _encode_queries_batch(self, queries: List[KBQuery]):
        pass


class RecencyEncoding(ReEncode):
    raise NotImplementedError

class BoostEncoding(ReEncode):
    raise NotImplementedError

class DimensionalityReduction(ReEncode):
    raise NotImplementedError


class PipelineEncoder(Encoder):

    def __init__(self, encoding_steps: List[Union[Encoder, ReEncode]], **kwargs):
        super().__init__(**kwargs)
        if len(encoding_steps) == 0:
            raise ValueError("Must provide at least one encoding step")

        if not isinstance(encoding_steps[0], Encoder):
            raise ValueError("First step must be an Encoder")

        self.encoding_steps = encoding_steps

    def _encode_documents_batch(self, documents: List[KBEncodedDocChunk]):
        for step in self.encoding_steps:
            # Each step is editing the encoded_chunks in place
            step.encode_documents(documents)

    def _encode_queries_batch(self, queries: List[KBQuery]):
        for step in self.encoding_steps:
            # Each step is editing the kb_queries in place
            step.encode_queries(queries)



