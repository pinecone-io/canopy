import mmh3
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from typing import Union, List

from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder


class StubDenseEncoder(BaseDenseEncoder):

    """
    Bag-of-words encoder that uses a random projection matrix to
    project sparse vectors to dense vectors.
    uses Johnson–Lindenstrauss lemma to project BOW sparse vectors to dense vectors.
    """

    def __init__(self,
                 dimension: int = 128,
                 vocab_size: int = 2 ** 20,
                 seed: int = 42):
        self.input_dim = vocab_size
        self.dimension = dimension
        rng = np.random.default_rng(seed)
        self.random_matrix = rng.standard_normal((self.input_dim, self.dimension))

    def _text_to_sparse_vector(self, text: str) -> csr_matrix:
        words = text.split()
        word_counts = defaultdict(int)
        for word in words:
            hashed_word = mmh3.hash(word) % self.input_dim
            word_counts[hashed_word] += 1

        indices = list(word_counts.keys())
        values = list(word_counts.values())
        sparse_vector = csr_matrix((values, (np.zeros_like(indices), indices)),
                                   shape=(1, self.input_dim))

        return sparse_vector

    def _encode_text(self, text: str) -> List[float]:
        sparse_vector = self._text_to_sparse_vector(text)
        projected_embedding = sparse_vector.dot(self.random_matrix).flatten()
        return list(projected_embedding / np.linalg.norm(projected_embedding))

    def encode_documents(self,
                         texts: Union[str, List[str]]
                         ) -> Union[List[float], List[List[float]]]:
        return self._encode(texts)

    def encode_queries(self,
                       texts: Union[str, List[str]]
                       ) -> Union[List[float], List[List[float]]]:
        return self._encode(texts)

    def _encode(self,
                texts: Union[str, List[str]]
                ) -> Union[List[float], List[List[float]]]:
        if isinstance(texts, str):
            return self._encode_text(texts)
        else:
            return [self._encode_text(text) for text in texts]
