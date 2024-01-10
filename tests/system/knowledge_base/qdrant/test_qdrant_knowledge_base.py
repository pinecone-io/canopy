import copy
import random

import pytest
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from canopy.knowledge_base.chunker.base import Chunker

from canopy.knowledge_base.knowledge_base import KnowledgeBase
from canopy.knowledge_base.qdrant.qdrant_knowledge_base import (
    DENSE_VECTOR_NAME,
    QdrantConverter,
    QdrantKnowledgeBase,
)
from canopy.knowledge_base.qdrant.qdrant_knowledge_base import COLLECTION_NAME_PREFIX
from canopy.knowledge_base.models import DocumentWithScore
from canopy.knowledge_base.record_encoder.base import RecordEncoder
from canopy.knowledge_base.reranker.reranker import Reranker
from canopy.models.data_models import Query

from qdrant_client.qdrant_remote import QdrantRemote
from tests.system.knowledge_base.qdrant.common import (
    assert_chunks_in_collection,
    assert_ids_in_collection,
    assert_ids_not_in_collection,
    assert_num_points_in_collection,
    total_vectors_in_collection,
)
from tests.unit.stubs.stub_chunker import StubChunker
from tests.unit.stubs.stub_dense_encoder import StubDenseEncoder
from tests.unit.stubs.stub_record_encoder import StubRecordEncoder

load_dotenv()


def execute_and_assert_queries(knowledge_base: QdrantKnowledgeBase, chunks_to_query):
    queries = [Query(text=chunk.text, top_k=2) for chunk in chunks_to_query]

    query_results = knowledge_base.query(queries)

    assert len(query_results) == len(queries)

    for i, q_res in enumerate(query_results):
        assert queries[i].text == q_res.query
        assert len(q_res.documents) == 2
        q_res.documents[0].score = round(q_res.documents[0].score, 1)
        assert q_res.documents[0] == DocumentWithScore(
            id=chunks_to_query[i].id,
            text=chunks_to_query[i].text,
            metadata=chunks_to_query[i].metadata,
            source=chunks_to_query[i].source,
            score=1.0,
        ), (
            f"query {i} - expected: {chunks_to_query[i]}, "
            f"actual: {q_res.documents[0]}"
        )


def assert_query_metadata_filter(
    knowledge_base: KnowledgeBase,
    metadata_filter: dict,
    num_vectors_expected: int,
    top_k: int = 100,
):
    assert (
        top_k > num_vectors_expected
    ), "the test might return false positive if top_k is not > num_vectors_expected"
    query = Query(text="test", top_k=top_k, metadata_filter=metadata_filter)
    query_results = knowledge_base.query([query])
    assert len(query_results) == 1
    assert len(query_results[0].documents) == num_vectors_expected


def test_create_collection(collection_full_name, knowledge_base: QdrantKnowledgeBase):
    assert knowledge_base.collection_name == collection_full_name
    collection_info = knowledge_base._client.get_collection(collection_full_name)
    assert (
        collection_info.config.params.vectors[DENSE_VECTOR_NAME].size
        == knowledge_base._encoder.dimension
    )


def test_list_collections(collection_full_name, knowledge_base: QdrantKnowledgeBase):
    collections_list = knowledge_base.list_canopy_collections()

    assert len(collections_list) > 0
    for item in collections_list:
        assert COLLECTION_NAME_PREFIX in item

    assert collection_full_name in collections_list


def test_is_verify_connection_happy_path(knowledge_base):
    knowledge_base.verify_index_connection()


def test_init_with_context_engine_prefix(collection_full_name, chunker, encoder):
    kb = QdrantKnowledgeBase(
        collection_name=collection_full_name,
        record_encoder=encoder,
        chunker=chunker,
    )
    assert kb.collection_name == collection_full_name


def test_upsert_happy_path(
    knowledge_base: QdrantKnowledgeBase, documents, encoded_chunks
):
    knowledge_base.upsert(documents)

    assert_num_points_in_collection(knowledge_base, len(encoded_chunks))
    assert_chunks_in_collection(knowledge_base, encoded_chunks)


@pytest.mark.parametrize("key", ["document_id", "text", "source", "chunk_id"])
def test_upsert_forbidden_metadata(knowledge_base, documents, key):
    doc = random.choice(documents)
    doc.metadata[key] = "bla"

    with pytest.raises(ValueError) as e:
        knowledge_base.upsert(documents)

    assert "reserved metadata keys" in str(e.value)
    assert doc.id in str(e.value)
    assert key in str(e.value)


def test_query(knowledge_base, encoded_chunks):
    execute_and_assert_queries(knowledge_base, encoded_chunks)


def test_query_with_metadata_filter(knowledge_base):
    if not isinstance(knowledge_base._client._client, QdrantRemote):
        pytest.skip(
            "Dict filter is not supported for QdrantLocal"
            "Use qdrant_client.models.Filter instead"
        )

    assert_query_metadata_filter(
        knowledge_base,
        {
            "must": [
                {"key": "my-key", "match": {"value": "value-1"}},
            ]
        },
        2,
    )


def test_delete_documents(knowledge_base: QdrantKnowledgeBase, encoded_chunks):
    chunk_ids = [QdrantConverter.convert_id(chunk.id) for chunk in encoded_chunks[-4:]]
    doc_ids = set(doc.document_id for doc in encoded_chunks[-4:])

    assert_ids_in_collection(knowledge_base, chunk_ids)

    before_vector_cnt = total_vectors_in_collection(knowledge_base)

    knowledge_base.delete(document_ids=list(doc_ids))

    assert_num_points_in_collection(knowledge_base, before_vector_cnt - len(chunk_ids))
    assert_ids_not_in_collection(knowledge_base, chunk_ids)


def test_update_documents(encoder, documents, encoded_chunks, knowledge_base):
    # chunker/kb that produces fewer chunks per doc
    chunker = StubChunker(num_chunks_per_doc=1)

    docs = documents[:2]
    doc_ids = [doc.id for doc in docs]
    chunk_ids = [
        QdrantConverter.convert_id(chunk.id)
        for chunk in encoded_chunks
        if chunk.document_id in doc_ids
    ]

    assert_ids_in_collection(knowledge_base, chunk_ids)

    docs[0].metadata["new_key"] = "new_value"
    knowledge_base.upsert(docs)

    updated_chunks = encoder.encode_documents(chunker.chunk_documents(docs))
    expected_chunks = [QdrantConverter.convert_id(chunk.id) for chunk in updated_chunks]
    assert_chunks_in_collection(knowledge_base, updated_chunks)

    unexpected_chunks = [
        QdrantConverter.convert_id(c_id)
        for c_id in chunk_ids
        if c_id not in expected_chunks
    ]
    assert len(unexpected_chunks) > 0, "bug in the test itself"

    assert_ids_not_in_collection(knowledge_base, unexpected_chunks)


def test_upsert_large_list_happy_path(
    knowledge_base, documents_large, encoded_chunks_large
):
    knowledge_base.upsert(documents_large)

    chunks_for_validation = encoded_chunks_large[:10] + encoded_chunks_large[-10:]
    assert_ids_in_collection(
        knowledge_base,
        [QdrantConverter.convert_id(chunk.id) for chunk in chunks_for_validation],
    )


def test_delete_large_df_happy_path(
    knowledge_base, documents_large, encoded_chunks_large
):
    knowledge_base.delete([doc.id for doc in documents_large])

    chunks_for_validation = encoded_chunks_large[:10] + encoded_chunks_large[-10:]
    assert_ids_not_in_collection(
        knowledge_base,
        [QdrantConverter.convert_id(chunk.id) for chunk in chunks_for_validation],
    )


def test_upsert_documents_with_datetime_metadata(
    knowledge_base, documents_with_datetime_metadata, datetime_metadata_encoded_chunks
):
    knowledge_base.upsert(documents_with_datetime_metadata)

    assert_ids_in_collection(
        knowledge_base,
        [
            QdrantConverter.convert_id(chunk.id)
            for chunk in datetime_metadata_encoded_chunks
        ],
    )


def test_query_edge_case_documents(knowledge_base, datetime_metadata_encoded_chunks):
    execute_and_assert_queries(knowledge_base, datetime_metadata_encoded_chunks)


def test_create_existing_collection(collection_full_name, knowledge_base):
    with pytest.raises(RuntimeError) as e:
        knowledge_base.create_canopy_collection()

    assert f"Collection {collection_full_name} already exists" in str(e.value)


def test_kb_non_existing_collection(knowledge_base):
    kb = copy.copy(knowledge_base)

    kb._collection_name = f"{COLLECTION_NAME_PREFIX}non-existing-collection"

    with pytest.raises(RuntimeError) as e:
        kb.verify_index_connection()
    expected_msg = (
        f"Collection {COLLECTION_NAME_PREFIX}non-existing-collection does not exist!"
    )
    assert expected_msg in str(e.value)


def test_init_defaults(collection_name, collection_full_name):
    new_kb = QdrantKnowledgeBase(collection_name)
    assert isinstance(new_kb._client, QdrantClient)
    assert new_kb.collection_name == collection_full_name
    assert isinstance(new_kb._chunker, Chunker)
    assert isinstance(
        new_kb._chunker, QdrantKnowledgeBase._DEFAULT_COMPONENTS["chunker"]
    )
    assert isinstance(new_kb._encoder, RecordEncoder)
    assert isinstance(
        new_kb._encoder, QdrantKnowledgeBase._DEFAULT_COMPONENTS["record_encoder"]
    )
    assert isinstance(new_kb._reranker, Reranker)
    assert isinstance(new_kb._reranker, KnowledgeBase._DEFAULT_COMPONENTS["reranker"])


def test_init_defaults_with_override(knowledge_base, chunker):
    collection_name = knowledge_base.collection_name
    new_kb = QdrantKnowledgeBase(collection_name=collection_name, chunker=chunker)
    assert isinstance(new_kb._client, QdrantClient)
    assert new_kb.collection_name == collection_name
    assert isinstance(new_kb._chunker, Chunker)
    assert isinstance(new_kb._chunker, StubChunker)
    assert new_kb._chunker is chunker
    assert isinstance(new_kb._encoder, RecordEncoder)
    assert isinstance(
        new_kb._encoder, KnowledgeBase._DEFAULT_COMPONENTS["record_encoder"]
    )
    assert isinstance(new_kb._reranker, Reranker)
    assert isinstance(new_kb._reranker, KnowledgeBase._DEFAULT_COMPONENTS["reranker"])


def test_init_raise_wrong_type(knowledge_base, chunker):
    collection_name = knowledge_base.collection_name
    with pytest.raises(TypeError) as e:
        QdrantKnowledgeBase(
            collection_name=collection_name,
            record_encoder=chunker,
        )

    assert "record_encoder must be an instance of RecordEncoder" in str(e.value)


def test_create_with_collection_encoder_dimension_none(collection_name, chunker):
    encoder = StubRecordEncoder(StubDenseEncoder(dimension=3))
    encoder._dense_encoder.dimension = None
    with pytest.raises(RuntimeError) as e:
        kb = QdrantKnowledgeBase(
            collection_name=collection_name,
            record_encoder=encoder,
            chunker=chunker,
        )
        kb.create_canopy_collection()

    assert "failed to infer" in str(e.value)
    assert "dimension" in str(e.value)
    assert f"{encoder.__class__.__name__} does not support" in str(e.value)
