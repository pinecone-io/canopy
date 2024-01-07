from typing import Dict, List, Any
import uuid
from qdrant_client import models
from canopy.knowledge_base.models import (
    KBDocChunkWithScore,
    KBEncodedDocChunk,
    VectorValues,
)
from pinecone_text.sparse import SparseVector

from canopy.knowledge_base.qdrant.constants import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    UUID_NAMESPACE,
)


class QdrantConverter:
    @staticmethod
    def convert_id(_id: str) -> str:
        """
        Converts any string into a UUID-like format in a deterministic way.

        Qdrant does not accept any string as an id, so an internal id has to be
        generated for each point. This is a deterministic way of doing so.
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
                vector[DENSE_VECTOR] = dense_vector

            if sparse_vector:
                vector[SPARSE_VECTOR] = models.SparseVector(
                    indices=sparse_vector["indices"],  # type: ignore
                    values=sparse_vector["values"],  # type: ignore
                )
            points.append(
                models.PointStruct(
                    id=QdrantConverter.convert_id(_id),
                    vector=vector,
                    payload={**record["metadata"], "id": _id},
                )
            )
        return points

    @staticmethod
    def scored_point_to_scored_doc(
        scored_point: models.ScoredPoint,
    ) -> "KBDocChunkWithScore":
        metadata: Dict[str, Any] = scored_point.payload  # type: ignore
        _id = scored_point.payload.pop("id")  # type: ignore
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
