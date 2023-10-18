import mmh3
import numpy as np
from collections import defaultdict
from typing import Union, List

from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder


class StubDenseEncoder(BaseDenseEncoder):
    """
    Bag-of-words encoder that uses a random projection matrix to
    project sparse vectors to dense vectors.
    uses Johnson–Lindenstrauss lemma to project BOW sparse vectors to dense vectors.
    """

    def __init__(self,
                 dimension: int = 8,
                 vocab_size: int = 2 ** 12):
        self.input_dim = vocab_size
        self.dimension = dimension

    def _text_to_word_counts(self, text: str) -> defaultdict:
        words = text.split()
        word_counts = defaultdict(int)
        for word in words:
            hashed_word = mmh3.hash(word) % self.input_dim
            word_counts[hashed_word] += 1
        return word_counts

    def _encode_text(self, text: str) -> List[float]:
        word_counts = self._text_to_word_counts(text)

        # This will hold the result of word_counts * random_matrix
        projected_embedding = np.zeros(self.dimension, dtype=np.float32)

        for hashed_word, count in word_counts.items():
            rng = np.random.default_rng(hashed_word)
            # Seed the RNG with the hashed word index for consistency
            random_vector = rng.standard_normal(self.dimension)
            projected_embedding += count * random_vector

        projected_embedding = projected_embedding.astype(np.float32)
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
