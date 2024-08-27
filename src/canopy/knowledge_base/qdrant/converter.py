from copy import deepcopy
from typing import Dict, List, Any, Tuple, Union
import uuid
from canopy.knowledge_base.models import (
    KBDocChunkWithScore,
    KBEncodedDocChunk,
    KBQuery,
    VectorValues,
)
from pinecone_text.sparse import SparseVector

try:
    from qdrant_client import models
except ImportError:
    pass

from canopy.knowledge_base.qdrant.constants import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    UUID_NAMESPACE,
)


class QdrantConverter:
    @staticmethod
    def convert_id(_id: str) -> str:
        """
        Converts any string into a UUID string based on a seed.

        Qdrant accepts UUID strings and unsigned integers as point ID.
        We use a seed to convert each string into a UUID string deterministically.
        This allows us to overwrite the same point with the original ID.
        """
        return str(uuid.uuid5(uuid.UUID(UUID_NAMESPACE), _id))

    @staticmethod
    def encoded_docs_to_points(
        encoded_docs: List[KBEncodedDocChunk],
    ) -> "List[models.PointStruct]":
        points = []
        for doc in encoded_docs:
            record = doc.to_db_record()
            _id: str = record.pop("id")
            dense_vector: VectorValues = record.pop("values", None)
            sparse_vector: SparseVector = record.pop("sparse_values", None)

            vector: Dict[str, models.Vector] = {}

            if dense_vector:
                vector[DENSE_VECTOR_NAME] = dense_vector

            if sparse_vector:
                vector[SPARSE_VECTOR_NAME] = models.SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"],
                )

            points.append(
                models.PointStruct(
                    id=QdrantConverter.convert_id(_id),
                    vector=vector,
                    payload={**record["metadata"], "chunk_id": _id},
                )
            )
        return points

    @staticmethod
    def scored_point_to_scored_doc(
        scored_point,
    ) -> "KBDocChunkWithScore":
        metadata: Dict[str, Any] = deepcopy(scored_point.payload or {})
        _id = metadata.pop("chunk_id")
        text = metadata.pop("text", "")
        document_id = metadata.pop("document_id")
        return KBDocChunkWithScore(
            id=_id,
            text=text,
            document_id=document_id,
            score=scored_point.score,
            source=metadata.pop("source", ""),
            metadata=metadata,
        )

    @staticmethod
    def kb_query_to_search_vector(
        query: KBQuery,
    ) -> "Tuple[Union[List[float], models.SparseVector], str]":
        # Use dense vector if available, otherwise use sparse vector
        if query.values:
            return query.values, DENSE_VECTOR_NAME
        elif query.sparse_values:
            return models.SparseVector(
                indices=query.sparse_values["indices"],
                values=query.sparse_values["values"],
            ), SPARSE_VECTOR_NAME
        else:
            raise ValueError("Query should have either dense or sparse vector.")
