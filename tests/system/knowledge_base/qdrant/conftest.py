import pytest
from canopy.knowledge_base.qdrant.constants import COLLECTION_NAME_PREFIX
from canopy.knowledge_base.qdrant.qdrant_knowledge_base import QdrantKnowledgeBase
from canopy.models.data_models import Document
from tests.system.knowledge_base.test_knowledge_base import _generate_text
from tests.unit.stubs.stub_chunker import StubChunker
from tests.unit.stubs.stub_dense_encoder import StubDenseEncoder
from tests.unit.stubs.stub_record_encoder import StubRecordEncoder
from tests.util import create_system_tests_index_name


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

@pytest.fixture(scope="module", autouse=True)
def teardown_knowledge_base(collection_full_name, knowledge_base):
    yield

    knowledge_base._client.delete_collection(collection_full_name)
    knowledge_base.close()


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