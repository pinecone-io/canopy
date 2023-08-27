import hashlib
import numpy as np
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

    def consistent_embedding(self, text: str) -> List[float]:
        # consistent embedding function that project each text to a unique angle
        embedding = []
        for i in range(self.dimension):
            sha256_hash = hashlib.sha256((text + str(i)).encode()).hexdigest()
            int_value = int(sha256_hash, 16)
            embedding.append(int_value / float(1 << 256))

        l2_norm = np.linalg.norm(embedding)
        normalized_embedding = [float(value / l2_norm) for value in embedding]

        return normalized_embedding

    def _encode(self,
                texts: Union[str, List[str]]
                ) -> Union[List[float], List[List[float]]]:
        if isinstance(texts, str):
            return self.consistent_embedding(texts)
        else:
            return [self.consistent_embedding(text) for text in texts]
