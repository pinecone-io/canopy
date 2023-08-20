import hashlib
from typing import Union, List

from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder


class StubDenseEncoder(BaseDenseEncoder):

    def __init__(self, dimension: int = 3):
        self.dimension = dimension

    def encode_documents(self,
                         texts: Union[str, List[str]]
                         ) -> Union[List[float], List[List[float]]]:
        return self._encode(texts)

    def encode_queries(self,
                       texts: Union[str, List[str]]
                       ) -> Union[List[float], List[List[float]]]:
        return self._encode(texts)

    @staticmethod
    def consistent_hash_float(text: str) -> float:
        sha256_hash = hashlib.sha256(text.encode()).hexdigest()
        int_value = int(sha256_hash, 16)
        # Normalize the integer value to a float in [0, 1)
        return int_value / float(1 << 256)

    def _encode(self,
                texts: Union[str, List[str]]
                ) -> Union[List[float], List[List[float]]]:
        if isinstance(texts, str):
            return [self.consistent_hash_float(texts)] * self.dimension
        else:
            return [[self.consistent_hash_float(t)] * self.dimension for t in texts]
