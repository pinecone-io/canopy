import pytest
import pinecone
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from context_engine.knoweldge_base import KnowledgeBase
from context_engine.knoweldge_base.models import DocumentWithScore
from context_engine.models.data_models import Document, Query
from tests.unit.stubs.stub_record_encoder import StubRecordEncoder
from tests.unit.stubs.stub_dense_encoder import StubDenseEncoder
from tests.unit.stubs.stub_chunker import StubChunker

load_dotenv()


class TestKnowledgeBase:

    @staticmethod
    @pytest.fixture(scope="class")
    def chunker():
        return StubChunker(num_chunks_per_doc=2)

    @staticmethod
    @pytest.fixture(scope="class")
    def encoder():
        return StubRecordEncoder(
            StubDenseEncoder(dimension=3))

    @staticmethod
    @pytest.fixture(scope="class", autouse=True)
    def knowledge_base(chunker, encoder):
        kb = KnowledgeBase(index_name="kb-integration-test",
                           encoder=encoder,
                           chunker=chunker)
        pinecone.init()
        if kb._index_name in pinecone.list_indexes():
            pinecone.delete_index(kb._index_name)

        kb.create_index()
        kb.connect()
        return kb

    @staticmethod
    @pytest.fixture(scope="class", autouse=True)
    def teardown_knowledge_base(knowledge_base):
        yield

        index_name = knowledge_base._index_name

        pinecone.init()
        if index_name in pinecone.list_indexes():
            pinecone.delete_index(index_name)

    @staticmethod
    @pytest.fixture
    def documents():
        return [Document(id=f"doc_{i}",
                         text=f"Sample document {i}",
                         metadata={"test": i})
                for i in range(5)]

    @staticmethod
    @pytest.fixture
    def encoded_chunks(documents, chunker, encoder):
        chunks = chunker.chunk_documents(documents)
        return encoder.encode_documents(chunks)

    @staticmethod
    def test_create_index(knowledge_base):
        index_name = knowledge_base._index_name
        assert index_name in pinecone.list_indexes()
        assert index_name == "context-engine-kb-integration-test"
        assert knowledge_base._index.describe_index_stats()

    @staticmethod
    def test_connect_connected_kb(knowledge_base):
        knowledge_base.connect()
        assert knowledge_base._index.describe_index_stats()

    @staticmethod
    def test_connect_force_connected_kb(knowledge_base):
        knowledge_base.connect(force=True)
        assert knowledge_base._index.describe_index_stats()

    @staticmethod
    def test_connect_unconnected_kb_index_exist():
        kb = KnowledgeBase(index_name="kb-integration-test",
                           encoder=StubRecordEncoder(
                               StubDenseEncoder(dimension=3)),
                           chunker=StubChunker())
        kb.connect()
        assert kb._index.describe_index_stats()

    @staticmethod
    def test_connect_unconnected_kb_index_not_exist_raise():
        kb = KnowledgeBase(index_name="not-exist",
                           encoder=StubRecordEncoder(
                               StubDenseEncoder(dimension=3)),
                           chunker=StubChunker())
        with pytest.raises(RuntimeError):
            kb.connect()

    @staticmethod
    def test_upsert(knowledge_base, documents, encoded_chunks):
        knowledge_base.upsert(documents)

        assert TestKnowledgeBase.total_vectors_in_index(
            knowledge_base) == len(encoded_chunks)
        TestKnowledgeBase.assert_chunks_in_index(
            knowledge_base, encoded_chunks)

    @staticmethod
    def test_upsert_dataframe(knowledge_base, documents, chunker, encoder):
        for doc in documents:
            doc.id = doc.id + "_df"
            doc.text = doc.text + " of df"

        vec_before = TestKnowledgeBase.total_vectors_in_index(knowledge_base)

        df = pd.DataFrame([{"id": doc.id, "text": doc.text, "metadata": doc.metadata}
                           for doc in documents])
        knowledge_base.upsert_dataframe(df)

        chunks = chunker.chunk_documents(documents)
        encoded_chunks = encoder.encode_documents(chunks)

        vec_after = TestKnowledgeBase.total_vectors_in_index(knowledge_base)

        assert vec_after - vec_before == len(encoded_chunks)
        TestKnowledgeBase.assert_chunks_in_index(
            knowledge_base, encoded_chunks)

    @staticmethod
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

    @staticmethod
    def test_delete_documents(knowledge_base, encoded_chunks):
        chunk_ids = [chunk.id for chunk in encoded_chunks[-4:]]
        doc_ids = set(doc.document_id for doc in encoded_chunks[-4:])

        fetch_result = knowledge_base._index.fetch(chunk_ids)
        assert len(fetch_result["vectors"]) == len(chunk_ids)

        TestKnowledgeBase.assert_ids_in_index(knowledge_base, chunk_ids)

        before_vector_cnt = TestKnowledgeBase.total_vectors_in_index(
            knowledge_base)

        knowledge_base.delete(document_ids=list(doc_ids))

        after_vector_cnt = TestKnowledgeBase.total_vectors_in_index(
            knowledge_base)

        assert before_vector_cnt - after_vector_cnt == len(chunk_ids)

        TestKnowledgeBase.assert_ids_not_in_index(knowledge_base, chunk_ids)

    @staticmethod
    def test_update_documents(encoder, documents, encoded_chunks):
        # chunker/kb that produces less chunks per doc
        chunker = StubChunker(num_chunks_per_doc=1)
        kb = KnowledgeBase(index_name="kb-integration-test",
                           encoder=encoder,
                           chunker=chunker)
        kb.connect()

        docs = documents[:2]
        doc_ids = [doc.id for doc in docs]
        chunk_ids = [chunk.id for chunk in encoded_chunks
                     if chunk.document_id in doc_ids]

        TestKnowledgeBase.assert_ids_in_index(kb, chunk_ids)

        kb.upsert(docs)

        updated_chunks = chunker.chunk_documents(docs)
        expected_chunks = [chunk.id for chunk in updated_chunks]
        TestKnowledgeBase.assert_ids_in_index(kb, expected_chunks)

        unexpected_chunks = [c_id for c_id in chunk_ids
                             if c_id not in expected_chunks]
        assert len(unexpected_chunks) > 0, "bug in the test itself"
        TestKnowledgeBase.assert_ids_not_in_index(kb, unexpected_chunks)

    @staticmethod
    def test_delete_index(knowledge_base):
        knowledge_base.delete_index()

        assert knowledge_base._index_name not in pinecone.list_indexes()
        assert knowledge_base._index is None
        with pytest.raises(RuntimeError):
            knowledge_base.connect()

    @staticmethod
    def total_vectors_in_index(knowledge_base):
        return knowledge_base._index.describe_index_stats().total_vector_count

    @staticmethod
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

    @staticmethod
    def assert_ids_in_index(knowledge_base, ids):
        fetch_result = knowledge_base._index.fetch(ids=ids)["vectors"]
        assert len(fetch_result) == len(ids)

    @staticmethod
    def assert_ids_not_in_index(knowledge_base, ids):
        fetch_result = knowledge_base._index.fetch(ids=ids)["vectors"]
        assert len(fetch_result) == 0
