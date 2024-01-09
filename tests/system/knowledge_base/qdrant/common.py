import numpy as np
from canopy.knowledge_base.qdrant.constants import DENSE_VECTOR_NAME
from canopy.knowledge_base.qdrant.converter import QdrantConverter
from canopy.knowledge_base.qdrant.qdrant_knowledge_base import QdrantKnowledgeBase


def total_vectors_in_collection(knowledge_base: QdrantKnowledgeBase):
    return knowledge_base._client.count(knowledge_base.collection_name).count


def assert_chunks_in_collection(knowledge_base: QdrantKnowledgeBase, encoded_chunks):
    ids = [QdrantConverter.convert_id(c.id) for c in encoded_chunks]
    fetch_result = knowledge_base._client.retrieve(
        knowledge_base.collection_name, ids=ids, with_payload=True, with_vectors=True
    )
    points = {p.id: p for p in fetch_result}
    for chunk in encoded_chunks:
        id = QdrantConverter.convert_id(chunk.id)
        assert id in points
        point = points[id]
        assert np.allclose(
            point.vector[DENSE_VECTOR_NAME],
            np.array(chunk.values, dtype=np.float32),
            atol=1e-8,
        )

        assert point.payload["text"] == chunk.text
        assert point.payload["document_id"] == chunk.document_id
        assert point.payload["source"] == chunk.source
        for key, value in chunk.metadata.items():
            assert point.payload[key] == value


def assert_ids_in_collection(knowledge_base, ids):
    fetch_result = knowledge_base._client.retrieve(
        knowledge_base.collection_name,
        ids=ids,
    )
    assert len(fetch_result) == len(
        ids
    ), f"Expected {len(ids)} ids, got {len(fetch_result)}"


def assert_num_points_in_collection(knowledge_base, num_vectors):
    points_in_index = total_vectors_in_collection(knowledge_base)
    assert (
        points_in_index == num_vectors
    ), f"Expected {num_vectors} vectors in index, got {points_in_index}"


def assert_ids_not_in_collection(knowledge_base, ids):
    fetch_result = knowledge_base._client.retrieve(
        knowledge_base.collection_name,
        ids=ids,
    )
    assert len(fetch_result) == 0, f"Found {len(fetch_result)} unexpected ids"