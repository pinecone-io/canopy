import random

import pytest

from canopy.knowledge_base.knowledge_base import KnowledgeBase
from canopy.knowledge_base.qdrant.qdrant_knowledge_base import (
    QdrantConverter,
    QdrantKnowledgeBase,
)
from tests.system.knowledge_base.qdrant.common import (
    assert_chunks_in_collection,
    assert_ids_in_collection,
    assert_ids_not_in_collection,
    assert_num_points_in_collection,
    total_vectors_in_collection,
)
from canopy.knowledge_base.models import DocumentWithScore
from canopy.models.data_models import Query
from tests.unit import random_words
from tests.unit.stubs.stub_chunker import StubChunker

qdrant_client = pytest.importorskip(
    "qdrant_client", reason="'qdrant_client' is not installed"
)


async def execute_and_assert_queries(
    knowledge_base: QdrantKnowledgeBase, chunks_to_query
):
    queries = [Query(text=chunk.text, top_k=2) for chunk in chunks_to_query]

    query_results = await knowledge_base.aquery(queries)

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


async def assert_query_metadata_filter(
    knowledge_base: KnowledgeBase,
    metadata_filter: dict,
    num_vectors_expected: int,
    top_k: int = 100,
):
    assert (
        top_k > num_vectors_expected
    ), "the test might return false positive if top_k is not > num_vectors_expected"

    query = Query(text="test", top_k=top_k, metadata_filter=metadata_filter)
    query_results = await knowledge_base.aquery([query])
    assert len(query_results) == 1
    assert len(query_results[0].documents) == num_vectors_expected


def _generate_text(num_words: int):
    return " ".join(random.choices(random_words, k=num_words))


@pytest.mark.asyncio
async def test_upsert_happy_path(
    knowledge_base: QdrantKnowledgeBase, documents, encoded_chunks
):
    await knowledge_base.aupsert(documents)
    assert_num_points_in_collection(knowledge_base, len(encoded_chunks))
    assert_chunks_in_collection(knowledge_base, encoded_chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["document_id", "text", "source"])
async def test_upsert_forbidden_metadata(knowledge_base, documents, key):
    doc = random.choice(documents)
    doc.metadata[key] = "bla"

    with pytest.raises(ValueError) as e:
        await knowledge_base.aupsert(documents)

    assert "reserved metadata keys" in str(e.value)
    assert doc.id in str(e.value)
    assert key in str(e.value)


@pytest.mark.asyncio
async def test_query(knowledge_base, encoded_chunks):
    await execute_and_assert_queries(knowledge_base, encoded_chunks)


@pytest.mark.asyncio
async def test_query_with_metadata_filter(knowledge_base):
    if knowledge_base._async_client is None or not isinstance(
        knowledge_base._async_client._client,
        qdrant_client.async_qdrant_remote.AsyncQdrantRemote,  # noqa: E501
    ):
        pytest.skip(
            "Dict filter is not supported for QdrantLocal"
            "Use qdrant_client.models.Filter instead"
        )

    await assert_query_metadata_filter(
        knowledge_base,
        {
            "must": [
                {"key": "my-key", "match": {"value": "value-1"}},
            ]
        },
        2,
    )


@pytest.mark.asyncio
async def test_delete_documents(knowledge_base: QdrantKnowledgeBase, encoded_chunks):
    chunk_ids = [QdrantConverter.convert_id(chunk.id) for chunk in encoded_chunks[-4:]]
    doc_ids = set(doc.document_id for doc in encoded_chunks[-4:])

    assert_ids_in_collection(knowledge_base, chunk_ids)

    before_vector_cnt = total_vectors_in_collection(knowledge_base)

    await knowledge_base.adelete(document_ids=list(doc_ids))

    assert_num_points_in_collection(knowledge_base, before_vector_cnt - len(chunk_ids))
    assert_ids_not_in_collection(knowledge_base, chunk_ids)


@pytest.mark.asyncio
async def test_update_documents(encoder, documents, encoded_chunks, knowledge_base):
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
    await knowledge_base.aupsert(docs)

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


@pytest.mark.asyncio
async def test_upsert_large_list_happy_path(
    knowledge_base, documents_large, encoded_chunks_large
):
    await knowledge_base.aupsert(documents_large)

    chunks_for_validation = encoded_chunks_large[:10] + encoded_chunks_large[-10:]
    assert_ids_in_collection(
        knowledge_base,
        [QdrantConverter.convert_id(chunk.id) for chunk in chunks_for_validation],
    )


@pytest.mark.asyncio
async def test_delete_large_df_happy_path(
    knowledge_base, documents_large, encoded_chunks_large
):
    await knowledge_base.adelete([doc.id for doc in documents_large])

    chunks_for_validation = encoded_chunks_large[:10] + encoded_chunks_large[-10:]
    assert_ids_not_in_collection(
        knowledge_base,
        [QdrantConverter.convert_id(chunk.id) for chunk in chunks_for_validation],
    )


@pytest.mark.asyncio
async def test_upsert_documents_with_datetime_metadata(
    knowledge_base, documents_with_datetime_metadata, datetime_metadata_encoded_chunks
):
    await knowledge_base.aupsert(documents_with_datetime_metadata)

    assert_ids_in_collection(
        knowledge_base,
        [
            QdrantConverter.convert_id(chunk.id)
            for chunk in datetime_metadata_encoded_chunks
        ],
    )


@pytest.mark.asyncio
async def test_query_edge_case_documents(
    knowledge_base, datetime_metadata_encoded_chunks
):
    await execute_and_assert_queries(knowledge_base, datetime_metadata_encoded_chunks)
