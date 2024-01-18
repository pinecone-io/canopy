import random
from typing import Dict, Any

import pytest
import numpy as np
from pinecone import Index, Pinecone
from tenacity import (
    retry,
    stop_after_delay,
    wait_fixed,
    wait_chain,
)

from canopy.knowledge_base import KnowledgeBase
from canopy.knowledge_base.chunker import Chunker
from canopy.knowledge_base.knowledge_base import (INDEX_NAME_PREFIX,
                                                  list_canopy_indexes,
                                                  _get_global_client)
from canopy.knowledge_base.models import DocumentWithScore
from canopy.knowledge_base.record_encoder import RecordEncoder
from canopy.knowledge_base.reranker import Reranker
from canopy.models.data_models import Document, Query
from tests.unit.stubs.stub_chunker import StubChunker
from tests.unit import random_words
from tests.unit.stubs.stub_dense_encoder import StubDenseEncoder
from tests.unit.stubs.stub_record_encoder import StubRecordEncoder
from tests.util import create_system_tests_index_name

PINECONE_API_KEY_ENV_VAR = "PINECONE_API_KEY"
RETRY_TIMEOUT = 120
FIRST_RETRY_WAIT = 10
RETRY_WAIT = 1


def retry_decorator():
    return retry(
        wait=wait_chain(*[wait_fixed(FIRST_RETRY_WAIT), wait_fixed(RETRY_WAIT)]),
        stop=stop_after_delay(RETRY_TIMEOUT),
    )


@pytest.fixture(scope="module")
def index_name(testrun_uid):
    return create_system_tests_index_name(testrun_uid)


@pytest.fixture(scope="module")
def index_full_name(index_name):
    return INDEX_NAME_PREFIX + index_name


@pytest.fixture(scope="module")
def chunker():
    return StubChunker(num_chunks_per_doc=2)


@pytest.fixture(scope="module")
def encoder():
    return StubRecordEncoder(
        StubDenseEncoder())


def try_create_canopy_index(kb: KnowledgeBase, init_params: Dict[str, Any]):
    kb.create_canopy_index(**init_params)


@pytest.fixture(scope="module", autouse=True)
def knowledge_base(index_full_name, index_name, chunker, encoder, create_index_params):
    kb = KnowledgeBase(index_name=index_name,
                       record_encoder=encoder,
                       chunker=chunker)

    if index_full_name in list_canopy_indexes():
        _get_global_client().delete_index(index_full_name)

    try_create_canopy_index(kb, create_index_params)

    return kb


def total_vectors_in_index(knowledge_base):
    return knowledge_base._index.describe_index_stats().total_vector_count


@retry_decorator()
def assert_chunks_in_index(knowledge_base, encoded_chunks):
    ids = [c.id for c in encoded_chunks]
    fetch_result = knowledge_base._index.fetch(ids=ids)["vectors"]
    for chunk in encoded_chunks:
        assert chunk.id in fetch_result
        fetched_chunk = fetch_result[chunk.id]
        assert np.allclose(fetched_chunk.values,
                           np.array(chunk.values, dtype=np.float32),
                           atol=1e-8)
        assert fetched_chunk.metadata["text"] == chunk.text
        assert fetched_chunk.metadata["document_id"] == chunk.document_id
        assert fetched_chunk.metadata["source"] == chunk.source
        for key, value in chunk.metadata.items():
            assert fetch_result[chunk.id].metadata[key] == value


@retry_decorator()
def assert_ids_in_index(knowledge_base, ids):
    fetch_result = knowledge_base._index.fetch(ids=ids)["vectors"]
    assert len(fetch_result) == len(ids), \
        f"Expected {len(ids)} ids, got {len(fetch_result.keys())}"


@retry_decorator()
def assert_num_vectors_in_index(knowledge_base, num_vectors):
    vectors_in_index = total_vectors_in_index(knowledge_base)
    assert vectors_in_index == num_vectors, \
        f"Expected {num_vectors} vectors in index, got {vectors_in_index}"


@retry_decorator()
def assert_ids_not_in_index(knowledge_base, ids):
    fetch_result = knowledge_base._index.fetch(ids=ids)["vectors"]
    assert len(fetch_result) == 0, f"Found unexpected ids: {len(fetch_result.keys())}"


@retry_decorator()
def execute_and_assert_queries(knowledge_base, chunks_to_query):
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
            score=1.0), \
            f"query {i} - expected: {chunks_to_query[i]}, " \
            f"actual: {q_res.documents}"


def assert_query_metadata_filter(knowledge_base: KnowledgeBase,
                                 metadata_filter: dict,
                                 num_vectors_expected: int,
                                 top_k: int = 100):
    assert top_k > num_vectors_expected, \
        "the test might return false positive if top_k is not > num_vectors_expected"
    query = Query(text="test", top_k=top_k, metadata_filter=metadata_filter)
    query_results = knowledge_base.query([query])
    assert len(query_results) == 1
    assert len(query_results[0].documents) == num_vectors_expected


@pytest.fixture(scope="module", autouse=True)
def teardown_knowledge_base(index_full_name, knowledge_base):
    yield
    if index_full_name in list_canopy_indexes():
        _get_global_client().delete_index(index_full_name)


def _generate_text(num_words: int):
    return " ".join(random.choices(random_words, k=num_words))


@pytest.fixture(scope="module")
def random_texts():
    return [_generate_text(10) for _ in range(5)]


@pytest.fixture
def documents(random_texts):
    return [Document(id=f"doc_{i}",
                     text=random_texts[i],
                     source=f"source_{i}",
                     metadata={"my-key": f"value-{i}"})
            for i in range(5)]


@pytest.fixture
def documents_large():
    return [Document(id=f"doc_{i}_large",
                     text=f"Sample document {i}",
                     metadata={"my-key-large": f"value-{i}"})
            for i in range(1000)]


@pytest.fixture
def encoded_chunks_large(documents_large, chunker, encoder):
    chunks = chunker.chunk_documents(documents_large)
    return encoder.encode_documents(chunks)


@pytest.fixture
def documents_with_datetime_metadata():
    return [Document(id="doc_1_metadata",
                     text="document with datetime metadata",
                     source="source_1",
                     metadata={"datetime": "2021-01-01T00:00:00Z",
                               "datetime_other_format": "January 1, 2021 00:00:00",
                               "datetime_other_format_2": "2210.03945"}),
            Document(id="2021-01-01T00:00:00Z",
                     text="id is datetime",
                     source="source_1")]


@pytest.fixture
def datetime_metadata_encoded_chunks(documents_with_datetime_metadata,
                                     chunker,
                                     encoder):
    chunks = chunker.chunk_documents(documents_with_datetime_metadata)
    return encoder.encode_documents(chunks)


@pytest.fixture
def encoded_chunks(documents, chunker, encoder):
    chunks = chunker.chunk_documents(documents)
    return encoder.encode_documents(chunks)


def test_create_index(index_full_name, knowledge_base):
    assert knowledge_base.index_name == index_full_name
    assert index_full_name in list_canopy_indexes()
    assert knowledge_base._index.describe_index_stats()
    index_description = _get_global_client().describe_index(index_full_name)
    assert index_description.dimension == knowledge_base._encoder.dimension


def test_list_indexes(index_full_name):
    index_list = list_canopy_indexes()

    assert len(index_list) > 0
    for item in index_list:
        assert INDEX_NAME_PREFIX in item

    assert index_full_name in index_list


def test_is_verify_index_connection_happy_path(knowledge_base):
    knowledge_base.verify_index_connection()


def test_init_with_context_engine_prefix(index_full_name, chunker, encoder):
    kb = KnowledgeBase(index_name=index_full_name,
                       record_encoder=encoder,
                       chunker=chunker)
    assert kb.index_name == index_full_name


def test_upsert_happy_path(knowledge_base, documents, encoded_chunks):
    knowledge_base.upsert(documents)

    assert_num_vectors_in_index(knowledge_base, len(encoded_chunks))
    assert_chunks_in_index(knowledge_base, encoded_chunks)


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


def test_query_with_metadata_filter(knowledge_base, encoded_chunks):
    assert_query_metadata_filter(knowledge_base, {"my-key": "value-1"}, 2)


def test_delete_documents(knowledge_base, encoded_chunks):
    chunk_ids = [chunk.id for chunk in encoded_chunks[-4:]]
    doc_ids = set(doc.document_id for doc in encoded_chunks[-4:])

    fetch_result = knowledge_base._index.fetch(chunk_ids)
    assert len(fetch_result["vectors"]) == len(chunk_ids)

    assert_ids_in_index(knowledge_base, chunk_ids)

    before_vector_cnt = total_vectors_in_index(
        knowledge_base)

    knowledge_base.delete(document_ids=list(doc_ids))

    assert_num_vectors_in_index(knowledge_base, before_vector_cnt - len(chunk_ids))
    assert_ids_not_in_index(knowledge_base, chunk_ids)


def test_update_documents(encoder,
                          documents,
                          encoded_chunks,
                          knowledge_base):
    index_name = knowledge_base._index_name

    # chunker/kb that produces fewer chunks per doc
    chunker = StubChunker(num_chunks_per_doc=1)
    kb = KnowledgeBase(index_name=index_name,
                       record_encoder=encoder,
                       chunker=chunker)
    kb.connect()
    docs = documents[:2]
    doc_ids = [doc.id for doc in docs]
    chunk_ids = [chunk.id for chunk in encoded_chunks
                 if chunk.document_id in doc_ids]

    assert_ids_in_index(kb, chunk_ids)

    docs[0].metadata["new_key"] = "new_value"
    kb.upsert(docs)

    updated_chunks = encoder.encode_documents(
        chunker.chunk_documents(docs)
    )
    expected_chunks = [chunk.id for chunk in updated_chunks]
    assert_chunks_in_index(kb, updated_chunks)

    if not knowledge_base._is_serverless_env():
        unexpected_chunks = [c_id for c_id in chunk_ids
                             if c_id not in expected_chunks]
        assert len(unexpected_chunks) > 0, "bug in the test itself"

        assert_ids_not_in_index(kb, unexpected_chunks)


def test_upsert_large_list_happy_path(knowledge_base,
                                      documents_large,
                                      encoded_chunks_large):
    knowledge_base.upsert(documents_large)

    chunks_for_validation = encoded_chunks_large[:10] + encoded_chunks_large[-10:]
    assert_ids_in_index(knowledge_base, [chunk.id
                                         for chunk in chunks_for_validation])


def test_delete_large_df_happy_path(knowledge_base,
                                    documents_large,
                                    encoded_chunks_large):
    knowledge_base.delete([doc.id for doc in documents_large])

    chunks_for_validation = encoded_chunks_large[:10] + encoded_chunks_large[-10:]
    assert_ids_not_in_index(knowledge_base, [chunk.id
                                             for chunk in chunks_for_validation])


def test_upsert_documents_with_datetime_metadata(knowledge_base,
                                                 documents_with_datetime_metadata,
                                                 datetime_metadata_encoded_chunks):
    knowledge_base.upsert(documents_with_datetime_metadata)

    assert_ids_in_index(knowledge_base, [chunk.id
                                         for chunk in datetime_metadata_encoded_chunks])


def test_query_edge_case_documents(knowledge_base,
                                   datetime_metadata_encoded_chunks):
    execute_and_assert_queries(knowledge_base, datetime_metadata_encoded_chunks)


def test_create_existing_index_no_connect(index_full_name, index_name):
    kb = KnowledgeBase(
        index_name=index_name,
        record_encoder=StubRecordEncoder(StubDenseEncoder(dimension=3)),
        chunker=StubChunker(num_chunks_per_doc=2))
    with pytest.raises(RuntimeError) as e:
        kb.create_canopy_index()

    assert f"Index {index_full_name} already exists" in str(e.value)


def test_kb_non_existing_index(index_name, chunker, encoder):
    kb = KnowledgeBase(index_name="non-existing-index",
                       record_encoder=encoder,
                       chunker=chunker)
    assert kb._index is None
    with pytest.raises(RuntimeError) as e:
        kb.connect()
    expected_msg = f"index {INDEX_NAME_PREFIX}non-existing-index does not exist"
    assert expected_msg in str(e.value)


@pytest.mark.parametrize("operation", ["upsert", "delete", "query",
                                       "verify_index_connection", "delete_index"])
def test_error_not_connected(operation, index_name):
    kb = KnowledgeBase(
        index_name=index_name,
        record_encoder=StubRecordEncoder(StubDenseEncoder(dimension=3)),
        chunker=StubChunker(num_chunks_per_doc=2))

    method = getattr(kb, operation)
    with pytest.raises(RuntimeError) as e:
        if operation == "verify_index_connection" or operation == "delete_index":
            method()
        else:
            method("dummy_input")
    assert "KnowledgeBase is not connected to index" in str(e.value)


def test_init_defaults(knowledge_base):
    index_name = knowledge_base.index_name
    new_kb = KnowledgeBase(index_name=index_name)
    new_kb.connect()
    assert isinstance(new_kb._index, Index)
    assert new_kb.index_name == index_name
    assert isinstance(new_kb._chunker, Chunker)
    assert isinstance(new_kb._chunker, KnowledgeBase._DEFAULT_COMPONENTS["chunker"])
    assert isinstance(new_kb._encoder, RecordEncoder)
    assert isinstance(new_kb._encoder,
                      KnowledgeBase._DEFAULT_COMPONENTS["record_encoder"])
    assert isinstance(new_kb._reranker, Reranker)
    assert isinstance(new_kb._reranker, KnowledgeBase._DEFAULT_COMPONENTS["reranker"])


def test_init_defaults_with_override(knowledge_base, chunker):
    index_name = knowledge_base.index_name
    new_kb = KnowledgeBase(index_name=index_name, chunker=chunker)
    new_kb.connect()
    assert isinstance(new_kb._index, Index)
    assert new_kb.index_name == index_name
    assert isinstance(new_kb._chunker, Chunker)
    assert isinstance(new_kb._chunker, StubChunker)
    assert new_kb._chunker is chunker
    assert isinstance(new_kb._encoder, RecordEncoder)
    assert isinstance(new_kb._encoder,
                      KnowledgeBase._DEFAULT_COMPONENTS["record_encoder"])
    assert isinstance(new_kb._reranker, Reranker)
    assert isinstance(new_kb._reranker, KnowledgeBase._DEFAULT_COMPONENTS["reranker"])


def test_init_raise_wrong_type(knowledge_base, chunker):
    index_name = knowledge_base.index_name
    with pytest.raises(TypeError) as e:
        KnowledgeBase(index_name=index_name, record_encoder=chunker)

    assert "record_encoder must be an instance of RecordEncoder" in str(e.value)


def test_delete_index_happy_path(knowledge_base):
    knowledge_base.delete_index()
    assert knowledge_base._index_name not in list_canopy_indexes()
    assert knowledge_base._index is None
    with pytest.raises(RuntimeError) as e:
        knowledge_base.delete(["doc_0"])
    assert "KnowledgeBase is not connected" in str(e.value)


def test_delete_index_for_non_existing(knowledge_base):
    with pytest.raises(RuntimeError) as e:
        knowledge_base.delete_index()

    assert "KnowledgeBase is not connected" in str(e.value)


def test_connect_after_delete(knowledge_base):
    with pytest.raises(RuntimeError) as e:
        knowledge_base.connect()

    assert "does not exist or was deleted" in str(e.value)


@pytest.mark.skip
def test_create_with_text_in_indexed_field_raise(index_name, chunker,
                                                 encoder):
    with pytest.raises(ValueError) as e:
        kb = KnowledgeBase(index_name=index_name,
                           record_encoder=encoder,
                           chunker=chunker)
        kb.create_canopy_index(indexed_fields=["id", "text", "metadata"])

    assert "The 'text' field cannot be used for metadata filtering" in str(e.value)


def test_create_with_index_encoder_dimension_none(index_name, chunker):
    encoder = StubRecordEncoder(StubDenseEncoder(dimension=3))
    encoder._dense_encoder.dimension = None
    with pytest.raises(RuntimeError) as e:
        kb = KnowledgeBase(index_name=index_name,
                           record_encoder=encoder,
                           chunker=chunker)
        kb.create_canopy_index()

    assert "failed to infer" in str(e.value)
    assert "dimension" in str(e.value)
    assert f"{encoder.__class__.__name__} does not support" in str(e.value)


# TODO: Add unit tests that verify that `pinecone.create_index()` is called with
#  correct `dimension` in all cases (inferred from encoder, directly passed, etc.)

# TODO: This test should be part of KnowledgeBase unit tests, which we don't have yet.
def test_create_encoder_err(index_name, chunker):
    class RaisesStubRecordEncoder(StubRecordEncoder):
        @property
        def dimension(self):
            raise ValueError("mock error")

    encoder = RaisesStubRecordEncoder(StubDenseEncoder(dimension=3))

    with pytest.raises(RuntimeError) as e:
        kb = KnowledgeBase(index_name=index_name,
                           record_encoder=encoder,
                           chunker=chunker)
        kb.create_canopy_index()

    assert "failed to infer" in str(e.value)
    assert "dimension" in str(e.value)
    assert "mock error" in str(e.value)
    assert encoder.__class__.__name__ in str(e.value)


@pytest.fixture
def unauthorized_pinecone_client():
    yield Pinecone(api_key="bad-key")


def test_create_bad_credentials(unauthorized_pinecone_client,
                                index_name, chunker, encoder):
    kb = KnowledgeBase(index_name=index_name,
                       record_encoder=encoder,
                       chunker=chunker,
                       pinecone_client=unauthorized_pinecone_client)

    with pytest.raises(RuntimeError) as e:
        kb.create_canopy_index()

    assert "Please check your credentials" in str(e.value)


def test_init_bad_credentials(unauthorized_pinecone_client,
                              index_name, chunker, encoder):
    kb = KnowledgeBase(index_name=index_name,
                       record_encoder=encoder,
                       chunker=chunker,
                       pinecone_client=unauthorized_pinecone_client)
    with pytest.raises(RuntimeError) as e:
        kb.connect()

    assert "Please check your credentials and try again" in str(e.value)
