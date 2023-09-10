import pytest
import pinecone
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from context_engine.knoweldge_base import KnowledgeBase
from context_engine.knoweldge_base.knowledge_base import INDEX_NAME_PREFIX
from context_engine.knoweldge_base.models import DocumentWithScore
from context_engine.models.data_models import Document, Query
from tests.unit.stubs.stub_record_encoder import StubRecordEncoder
from tests.unit.stubs.stub_dense_encoder import StubDenseEncoder
from tests.unit.stubs.stub_chunker import StubChunker

load_dotenv()


@pytest.mark.xdist_group(name="group1")
@pytest.fixture(scope="session")
def index_name(testrun_uid):
    today = datetime.today().strftime("%Y-%m-%d")
    return f"test-kb-{testrun_uid[-6:]}-{today}"


@pytest.fixture(scope="session")
def index_full_name(index_name):
    return INDEX_NAME_PREFIX + index_name


@pytest.fixture(scope="session")
def chunker():
    return StubChunker(num_chunks_per_doc=2)


@pytest.fixture(scope="session")
def encoder():
    return StubRecordEncoder(
        StubDenseEncoder(dimension=3))


@pytest.fixture(scope="session", autouse=True)
def knowledge_base(index_full_name, index_name, chunker, encoder):
    pinecone.init()
    if index_full_name in pinecone.list_indexes():
        pinecone.delete_index(index_full_name)

    KnowledgeBase.create_with_new_index(index_name=index_name,
                                        encoder=encoder,
                                        chunker=chunker)

    kb = KnowledgeBase(index_name=index_name,
                       encoder=encoder,
                       chunker=chunker)

    return kb


def total_vectors_in_index(knowledge_base):
    return knowledge_base._index.describe_index_stats().total_vector_count


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


def assert_ids_in_index(knowledge_base, ids):
    fetch_result = knowledge_base._index.fetch(ids=ids)["vectors"]
    assert len(fetch_result) == len(ids)


def assert_ids_not_in_index(knowledge_base, ids):
    fetch_result = knowledge_base._index.fetch(ids=ids)["vectors"]
    assert len(fetch_result) == 0


@pytest.fixture(scope="session", autouse=True)
def teardown_knowledge_base(knowledge_base):
    yield

    index_name = knowledge_base._index_name

    pinecone.init()
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)


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


def test_upsert(knowledge_base, documents, encoded_chunks):
    knowledge_base.upsert(documents)

    assert total_vectors_in_index(
        knowledge_base) == len(encoded_chunks)
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

    vec_after = total_vectors_in_index(knowledge_base)

    assert vec_after - vec_before == len(encoded_chunks)
    assert_chunks_in_index(knowledge_base, encoded_chunks)


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
        q_res.documents[0].score = round(q_res.documents[0].score, 6)
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

    after_vector_cnt = total_vectors_in_index(knowledge_base)

    assert before_vector_cnt - after_vector_cnt == len(chunk_ids)

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


def test_delete_index(knowledge_base):
    knowledge_base.delete_index()

    assert knowledge_base._index_name not in pinecone.list_indexes()
    assert knowledge_base._index is None
    with pytest.raises(RuntimeError) as e:
        knowledge_base.delete(["doc_0"])

    assert "index was deleted." in str(e.value)
