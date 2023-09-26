import os
import pytest
import pinecone
import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv
from datetime import datetime
from resin.knoweldge_base import KnowledgeBase
from resin.knoweldge_base.knowledge_base import INDEX_NAME_PREFIX
from resin.knoweldge_base.models import DocumentWithScore
from resin.models.data_models import Document, Query
from tests.unit.stubs.stub_record_encoder import StubRecordEncoder
from tests.unit.stubs.stub_dense_encoder import StubDenseEncoder
from tests.unit.stubs.stub_chunker import StubChunker


load_dotenv()

PINECONE_API_KEY_ENV_VAR = "PINECONE_API_KEY"


@pytest.fixture(scope="module")
def index_name(testrun_uid):
    today = datetime.today().strftime("%Y-%m-%d")
    return f"test-kb-{testrun_uid[-6:]}-{today}"


@pytest.fixture(scope="module")
def index_full_name(index_name):
    return INDEX_NAME_PREFIX + index_name


@pytest.fixture(scope="module")
def chunker():
    return StubChunker(num_chunks_per_doc=2)


@pytest.fixture(scope="module")
def encoder():
    return StubRecordEncoder(
        StubDenseEncoder(dimension=3))


@pytest.fixture(scope="module", autouse=True)
def knowledge_base(index_full_name, index_name, chunker, encoder):
    pinecone.init()
    if index_full_name in pinecone.list_indexes():
        pinecone.delete_index(index_full_name)

    KnowledgeBase.create_with_new_index(index_name=index_name,
                                        encoder=encoder,
                                        chunker=chunker)

    return KnowledgeBase(index_name=index_name,
                         encoder=encoder,
                         chunker=chunker)


def total_vectors_in_index(knowledge_base):
    return knowledge_base._index.describe_index_stats().total_vector_count


@retry(stop=stop_after_attempt(30), wait=wait_fixed(1))
def assert_chunks_in_index(knowledge_base, encoded_chunks):
    ids = [c.id for c in encoded_chunks]
    fetch_result = knowledge_base._index.fetch(ids=ids)["vectors"]
    for chunk in encoded_chunks:
        assert chunk.id in fetch_result
        assert np.allclose(fetch_result[chunk.id].values,
                           np.array(chunk.values, dtype=np.float32),
                           atol=1e-8)
        assert fetch_result[chunk.id].metadata["text"] == chunk.text
        assert fetch_result[chunk.id].metadata["document_id"] == chunk.document_id


@retry(stop=stop_after_attempt(30), wait=wait_fixed(1))
def assert_ids_in_index(knowledge_base, ids):
    fetch_result = knowledge_base._index.fetch(ids=ids)["vectors"]
    assert len(fetch_result) == len(ids)


@retry(stop=stop_after_attempt(30), wait=wait_fixed(1))
def assert_num_vectors_in_index(knowledge_base, num_vectors):
    assert total_vectors_in_index(knowledge_base) == num_vectors


@retry(stop=stop_after_attempt(30), wait=wait_fixed(1))
def assert_ids_not_in_index(knowledge_base, ids):
    fetch_result = knowledge_base._index.fetch(ids=ids)["vectors"]
    assert len(fetch_result) == 0


@pytest.fixture(scope="module", autouse=True)
def teardown_knowledge_base(index_full_name, knowledge_base):
    yield

    pinecone.init()
    if index_full_name in pinecone.list_indexes():
        pinecone.delete_index(index_full_name)


@pytest.fixture
def documents():
    return [Document(id=f"doc_{i}",
                     text=f"Sample document {i}",
                     metadata={"test": i})
            for i in range(5)]


@pytest.fixture
def encoded_chunks(documents, chunker, encoder):
    chunks = chunker.chunk_documents(documents)
    return encoder.encode_documents(chunks)


def test_create_index(index_full_name, knowledge_base):
    assert knowledge_base.index_name == index_full_name
    assert index_full_name in pinecone.list_indexes()
    assert index_full_name == index_full_name
    assert knowledge_base._index.describe_index_stats()


def test_is_verify_connection_health_happy_path(knowledge_base):
    knowledge_base.verify_connection_health()


def test_init_with_context_engine_prefix(index_full_name, chunker, encoder):
    kb = KnowledgeBase(index_name=index_full_name,
                       encoder=encoder,
                       chunker=chunker)
    assert kb.index_name == index_full_name


def test_upsert(knowledge_base, documents, encoded_chunks):
    knowledge_base.upsert(documents)

    assert_num_vectors_in_index(knowledge_base, len(encoded_chunks))
    assert_chunks_in_index(knowledge_base, encoded_chunks)


def test_upsert_dataframe(knowledge_base, documents, chunker, encoder):
    for doc in documents:
        doc.id = doc.id + "_df"
        doc.text = doc.text + " of df"

    vec_before = total_vectors_in_index(knowledge_base)

    df = pd.DataFrame([{"id": doc.id, "text": doc.text, "metadata": doc.metadata}
                       for doc in documents])
    knowledge_base.upsert_dataframe(df)

    chunks = chunker.chunk_documents(documents)
    encoded_chunks = encoder.encode_documents(chunks)


    assert_num_vectors_in_index(knowledge_base, vec_before + len(encoded_chunks))
    assert_chunks_in_index(knowledge_base, encoded_chunks)


def test_upsert_datafarme_with_wrong_schema(knowledge_base, documents):
    df = pd.DataFrame([{"id": doc.id, "text": doc.text, "md": doc.metadata}
                       for doc in documents])

    with pytest.raises(ValueError) as e:
        knowledge_base.upsert_dataframe(df)

    assert "Dataframe must contain the following columns" in str(e.value)


def test_query(knowledge_base, encoded_chunks):
    queries = [Query(text=encoded_chunks[0].text),
               Query(text=encoded_chunks[1].text, top_k=2)]
    query_results = knowledge_base.query(queries)

    assert len(query_results) == 2

    expected_top_k = [10, 2]
    expected_first_results = [DocumentWithScore(id=chunk.id,
                                                text=chunk.text,
                                                metadata=chunk.metadata,
                                                score=1.0)
                              for chunk in encoded_chunks[:2]]
    for i, q_res in enumerate(query_results):
        assert queries[i].text == q_res.query
        assert len(q_res.documents) == expected_top_k[i]
        q_res.documents[0].score = round(q_res.documents[0].score, 2)
        assert q_res.documents[0] == expected_first_results[i]


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


def test_update_documents(encoder, documents, encoded_chunks, knowledge_base):
    index_name = knowledge_base._index_name

    # chunker/kb that produces less chunks per doc
    chunker = StubChunker(num_chunks_per_doc=1)
    kb = KnowledgeBase(index_name=index_name,
                       encoder=encoder,
                       chunker=chunker)
    docs = documents[:2]
    doc_ids = [doc.id for doc in docs]
    chunk_ids = [chunk.id for chunk in encoded_chunks
                 if chunk.document_id in doc_ids]

    assert_ids_in_index(kb, chunk_ids)

    kb.upsert(docs)

    updated_chunks = chunker.chunk_documents(docs)
    expected_chunks = [chunk.id for chunk in updated_chunks]
    assert_ids_in_index(kb, expected_chunks)

    unexpected_chunks = [c_id for c_id in chunk_ids
                         if c_id not in expected_chunks]
    assert len(unexpected_chunks) > 0, "bug in the test itself"
    assert_ids_not_in_index(kb, unexpected_chunks)


def test_create_existing_index(index_full_name, index_name):
    with pytest.raises(RuntimeError) as e:
        KnowledgeBase.create_with_new_index(index_name=index_name,
                                            encoder=StubRecordEncoder(
                                                StubDenseEncoder(dimension=3)),
                                            chunker=StubChunker(num_chunks_per_doc=2))

    assert f"Index {index_full_name} already exists" in str(e.value)


def test_init_kb_non_existing_index(index_name, chunker, encoder):
    with pytest.raises(RuntimeError) as e:
        KnowledgeBase(index_name="non-existing-index",
                      encoder=encoder,
                      chunker=chunker)
    expected_msg = f"Index {INDEX_NAME_PREFIX}non-existing-index does not exist"
    assert expected_msg in str(e.value)


def test_delete_index_happy_path(knowledge_base):
    knowledge_base.delete_index()

    assert knowledge_base._index_name not in pinecone.list_indexes()
    assert knowledge_base._index is None
    with pytest.raises(RuntimeError) as e:
        knowledge_base.delete(["doc_0"])

    assert "index was deleted." in str(e.value)


def test_delete_index_for_non_existing(knowledge_base):
    with pytest.raises(RuntimeError) as e:
        knowledge_base.delete_index()

    assert "index was deleted." in str(e.value)


def test_verify_connection_health_raise_for_deleted_index(knowledge_base):
    with pytest.raises(RuntimeError) as e:
        knowledge_base.verify_connection_health()

    assert "index was deleted" in str(e.value)


def test_create_with_text_in_indexed_field_raise(index_name,
                                                 chunker,
                                                 encoder):
    with pytest.raises(ValueError) as e:
        KnowledgeBase.create_with_new_index(index_name=index_name,
                                            encoder=encoder,
                                            chunker=chunker,
                                            indexed_fields=["id", "text", "metadata"])

    assert "The 'text' field cannot be used for metadata filtering" in str(e.value)


def test_create_with_new_index_encoder_dimension_none(index_name, chunker):
    encoder = StubRecordEncoder(StubDenseEncoder(dimension=3))
    encoder._dense_encoder.dimension = None
    with pytest.raises(ValueError) as e:
        KnowledgeBase.create_with_new_index(index_name=index_name,
                                            encoder=encoder,
                                            chunker=chunker)

    assert "Could not infer dimension from encoder" in str(e.value)


@pytest.fixture
def set_bad_credentials():
    original_api_key = os.environ.get(PINECONE_API_KEY_ENV_VAR)

    os.environ[PINECONE_API_KEY_ENV_VAR] = "bad-key"

    yield

    # Restore the original API key after test execution
    os.environ[PINECONE_API_KEY_ENV_VAR] = original_api_key


def test_create_bad_credentials(set_bad_credentials, index_name, chunker, encoder):
    with pytest.raises(RuntimeError) as e:
        KnowledgeBase.create_with_new_index(index_name=index_name,
                                            encoder=encoder,
                                            chunker=chunker)

    assert "Please check your credentials" in str(e.value)


def test_init_bad_credentials(set_bad_credentials, index_name, chunker, encoder):
    with pytest.raises(RuntimeError) as e:
        KnowledgeBase(index_name=index_name,
                      encoder=encoder,
                      chunker=chunker)

    assert "Please check your credentials and try again" in str(e.value)
