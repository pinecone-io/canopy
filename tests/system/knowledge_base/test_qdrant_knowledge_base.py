import random

import numpy as np
import pytest
from dotenv import load_dotenv

from canopy.knowledge_base.knowledge_base import KnowledgeBase
from canopy.knowledge_base.qdrant.qdrant_knowledge_base import (
    DENSE_VECTOR,
    QdrantConverter,
    QdrantKnowledgeBase,
)
from canopy.knowledge_base.qdrant.qdrant_knowledge_base import COLLECTION_NAME_PREFIX
from canopy.knowledge_base.models import DocumentWithScore
from canopy.models.data_models import Document, Query
from tests.unit import random_words
from tests.unit.stubs.stub_chunker import StubChunker
from tests.unit.stubs.stub_dense_encoder import StubDenseEncoder
from tests.unit.stubs.stub_record_encoder import StubRecordEncoder
from tests.util import create_system_tests_index_name

load_dotenv()

PINECONE_API_KEY_ENV_VAR = "PINECONE_API_KEY"


@pytest.fixture(scope="module")
def collection_name(testrun_uid):
    return create_system_tests_index_name(testrun_uid)


@pytest.fixture(scope="module")
def collection_full_name(collection_name):
    return COLLECTION_NAME_PREFIX + collection_name


@pytest.fixture(scope="module")
def chunker():
    return StubChunker(num_chunks_per_doc=2)


@pytest.fixture(scope="module")
def encoder():
    return StubRecordEncoder(StubDenseEncoder())


@pytest.fixture(scope="module", autouse=True)
def knowledge_base(collection_name, chunker, encoder):
    kb = QdrantKnowledgeBase(
        collection_name=collection_name,
        record_encoder=encoder,
        chunker=chunker,
        location="http://localhost:6333",
    )
    kb.create_canopy_collection()

    return kb


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
            point.vector[DENSE_VECTOR],
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


@pytest.fixture(scope="module", autouse=True)
def teardown_knowledge_base(collection_full_name, knowledge_base):
    yield

    knowledge_base._client.delete_collection(collection_full_name)


def _generate_text(num_words: int):
    return " ".join(random.choices(random_words, k=num_words))


@pytest.fixture(scope="module")
def random_texts():
    return [_generate_text(10) for _ in range(5)]


@pytest.fixture
def documents(random_texts):
    return [
        Document(
            id=f"doc_{i}",
            text=random_texts[i],
            source=f"source_{i}",
            metadata={"my-key": f"value-{i}"},
        )
        for i in range(5)
    ]


@pytest.fixture
def documents_large():
    return [
        Document(
            id=f"doc_{i}_large",
            text=f"Sample document {i}",
            metadata={"my-key-large": f"value-{i}"},
        )
        for i in range(1000)
    ]


@pytest.fixture
def encoded_chunks_large(documents_large, chunker, encoder):
    chunks = chunker.chunk_documents(documents_large)
    return encoder.encode_documents(chunks)


@pytest.fixture
def documents_with_datetime_metadata():
    return [
        Document(
            id="doc_1_metadata",
            text="document with datetime metadata",
            source="source_1",
            metadata={
                "datetime": "2021-01-01T00:00:00Z",
                "datetime_other_format": "January 1, 2021 00:00:00",
                "datetime_other_format_2": "2210.03945",
            },
        ),
        Document(id="2021-01-01T00:00:00Z", text="id is datetime", source="source_1"),
    ]


@pytest.fixture
def datetime_metadata_encoded_chunks(
    documents_with_datetime_metadata, chunker, encoder
):
    chunks = chunker.chunk_documents(documents_with_datetime_metadata)
    return encoder.encode_documents(chunks)


@pytest.fixture
def encoded_chunks(documents, chunker, encoder):
    chunks = chunker.chunk_documents(documents)
    return encoder.encode_documents(chunks)


def test_create_index(collection_full_name, knowledge_base: QdrantKnowledgeBase):
    assert knowledge_base.collection_name == collection_full_name
    collection_info = knowledge_base._client.get_collection(collection_full_name)
    assert (
        collection_info.config.params.vectors[DENSE_VECTOR].size
        == knowledge_base._encoder.dimension
    )


def test_list_indexes(collection_full_name, knowledge_base: QdrantKnowledgeBase):
    collections_list = knowledge_base.list_canopy_collections()

    assert len(collections_list) > 0
    for item in collections_list:
        assert COLLECTION_NAME_PREFIX in item

    assert collection_full_name in collections_list


def test_is_verify_connection_happy_path(knowledge_base):
    knowledge_base.verify_index_connection()


def test_init_with_context_engine_prefix(collection_full_name, chunker, encoder):
    kb = QdrantKnowledgeBase(
        collection_name=collection_full_name, record_encoder=encoder, chunker=chunker
    )
    assert kb.collection_name == collection_full_name


def test_upsert_happy_path(
    knowledge_base: QdrantKnowledgeBase, documents, encoded_chunks
):
    knowledge_base.upsert(documents)

    assert_num_points_in_collection(knowledge_base, len(encoded_chunks))
    assert_chunks_in_collection(knowledge_base, encoded_chunks)


@pytest.mark.parametrize("key", ["document_id", "text", "source"])
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
    collection_name = knowledge_base.collection_name

    # chunker/kb that produces fewer chunks per doc
    chunker = StubChunker(num_chunks_per_doc=1)
    kb = QdrantKnowledgeBase(collection_name, record_encoder=encoder, chunker=chunker)
    docs = documents[:2]
    doc_ids = [doc.id for doc in docs]
    chunk_ids = [
        QdrantConverter.convert_id(chunk.id)
        for chunk in encoded_chunks
        if chunk.document_id in doc_ids
    ]

    assert_ids_in_collection(kb, chunk_ids)

    docs[0].metadata["new_key"] = "new_value"
    kb.upsert(docs)

    updated_chunks = encoder.encode_documents(chunker.chunk_documents(docs))
    expected_chunks = [QdrantConverter.convert_id(chunk.id) for chunk in updated_chunks]
    assert_chunks_in_collection(kb, updated_chunks)

    unexpected_chunks = [
        QdrantConverter.convert_id(c_id)
        for c_id in chunk_ids
        if c_id not in expected_chunks
    ]
    assert len(unexpected_chunks) > 0, "bug in the test itself"

    assert_ids_not_in_collection(kb, unexpected_chunks)


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


def test_create_existing_index_no_connect(collection_full_name, collection_name):
    kb = QdrantKnowledgeBase(
        collection_name,
        record_encoder=StubRecordEncoder(StubDenseEncoder(dimension=3)),
        chunker=StubChunker(num_chunks_per_doc=2),
    )
    with pytest.raises(RuntimeError) as e:
        kb.create_canopy_collection()

    assert f"Collection {collection_full_name} already exists" in str(e.value)


def test_kb_non_existing_index(chunker, encoder):
    kb = QdrantKnowledgeBase(
        "non-existing-collection", record_encoder=encoder, chunker=chunker
    )

    with pytest.raises(RuntimeError) as e:
        kb.verify_index_connection()
    expected_msg = (
        f"Collection {COLLECTION_NAME_PREFIX}non-existing-collection does not exist!"
    )
    assert expected_msg in str(e.value)


# def test_init_defaults(knowledge_base):
#     index_name = knowledge_base.index_name
#     new_kb = KnowledgeBase(index_name=index_name)
#     new_kb.connect()
#     assert isinstance(new_kb._index, pinecone.Index)
#     assert new_kb.index_name == index_name
#     assert isinstance(new_kb._chunker, Chunker)
#     assert isinstance(new_kb._chunker, KnowledgeBase._DEFAULT_COMPONENTS["chunker"])
#     assert isinstance(new_kb._encoder, RecordEncoder)
#     assert isinstance(new_kb._encoder,
#                       KnowledgeBase._DEFAULT_COMPONENTS["record_encoder"])
#     assert isinstance(new_kb._reranker, Reranker)
#     assert isinstance(new_kb._reranker, KnowledgeBase._DEFAULT_COMPONENTS["reranker"])


# def test_init_defaults_with_override(knowledge_base, chunker):
#     index_name = knowledge_base.index_name
#     new_kb = KnowledgeBase(index_name=index_name, chunker=chunker)
#     new_kb.connect()
#     assert isinstance(new_kb._index, pinecone.Index)
#     assert new_kb.index_name == index_name
#     assert isinstance(new_kb._chunker, Chunker)
#     assert isinstance(new_kb._chunker, StubChunker)
#     assert new_kb._chunker is chunker
#     assert isinstance(new_kb._encoder, RecordEncoder)
#     assert isinstance(new_kb._encoder,
#                       KnowledgeBase._DEFAULT_COMPONENTS["record_encoder"])
#     assert isinstance(new_kb._reranker, Reranker)
#     assert isinstance(new_kb._reranker, KnowledgeBase._DEFAULT_COMPONENTS["reranker"])


# def test_init_raise_wrong_type(knowledge_base, chunker):
#     index_name = knowledge_base.index_name
#     with pytest.raises(TypeError) as e:
#         KnowledgeBase(index_name=index_name, record_encoder=chunker)

#     assert "record_encoder must be an instance of RecordEncoder" in str(e.value)


# def test_delete_index_happy_path(knowledge_base):
#     knowledge_base.delete_index()

#     assert knowledge_base._index_name not in pinecone.list_indexes()
#     assert knowledge_base._index is None
#     with pytest.raises(RuntimeError) as e:
#         knowledge_base.delete(["doc_0"])
#     assert "KnowledgeBase is not connected" in str(e.value)


# def test_delete_index_for_non_existing(knowledge_base):
#     with pytest.raises(RuntimeError) as e:
#         knowledge_base.delete_index()

#     assert "KnowledgeBase is not connected" in str(e.value)


# def test_connect_after_delete(knowledge_base):
#     with pytest.raises(RuntimeError) as e:
#         knowledge_base.connect()

#     assert "does not exist or was deleted" in str(e.value)


# def test_create_with_text_in_indexed_field_raise(index_name,
#                                                  chunker,
#                                                  encoder):
#     with pytest.raises(ValueError) as e:
#         kb = KnowledgeBase(index_name=index_name,
#                            record_encoder=encoder,
#                            chunker=chunker)
#         kb.create_canopy_index(indexed_fields=["id", "text", "metadata"])

#     assert "The 'text' field cannot be used for metadata filtering" in str(e.value)


# def test_create_with_index_encoder_dimension_none(index_name, chunker):
#     encoder = StubRecordEncoder(StubDenseEncoder(dimension=3))
#     encoder._dense_encoder.dimension = None
#     with pytest.raises(RuntimeError) as e:
#         kb = KnowledgeBase(index_name=index_name,
#                            record_encoder=encoder,
#                            chunker=chunker)
#         kb.create_canopy_index()

#     assert "failed to infer" in str(e.value)
#     assert "dimension" in str(e.value)
#     assert f"{encoder.__class__.__name__} does not support" in str(e.value)


# # TODO: Add unit tests that verify that `pinecone.create_index()` is called with
# #  correct `dimension` in all cases (inferred from encoder, directly passed, etc.)

# # TODO: This test should be part of KnowledgeBase unit tests, which we don't have yet.
# def test_create_encoder_err(index_name, chunker):
#     class RaisesStubRecordEncoder(StubRecordEncoder):
#         @property
#         def dimension(self):
#             raise ValueError("mock error")

#     encoder = RaisesStubRecordEncoder(StubDenseEncoder(dimension=3))

#     with pytest.raises(RuntimeError) as e:
#         kb = KnowledgeBase(index_name=index_name,
#                            record_encoder=encoder,
#                            chunker=chunker)
#         kb.create_canopy_index()

#     assert "failed to infer" in str(e.value)
#     assert "dimension" in str(e.value)
#     assert "mock error" in str(e.value)
#     assert encoder.__class__.__name__ in str(e.value)


# @pytest.fixture
# def set_bad_credentials():
#     original_api_key = os.environ.get(PINECONE_API_KEY_ENV_VAR)

#     os.environ[PINECONE_API_KEY_ENV_VAR] = "bad-key"

#     yield

#     # Restore the original API key after test execution
#     os.environ[PINECONE_API_KEY_ENV_VAR] = original_api_key


# def test_create_bad_credentials(set_bad_credentials, index_name, chunker, encoder):
#     kb = KnowledgeBase(index_name=index_name,
#                        record_encoder=encoder,
#                        chunker=chunker)
#     with pytest.raises(RuntimeError) as e:
#         kb.create_canopy_index()

#     assert "Please check your credentials" in str(e.value)


# def test_init_bad_credentials(set_bad_credentials, index_name, chunker, encoder):
#     kb = KnowledgeBase(index_name=index_name,
#                        record_encoder=encoder,
#                        chunker=chunker)
#     with pytest.raises(RuntimeError) as e:
#         kb.connect()

#     assert "Please check your credentials and try again" in str(e.value)
