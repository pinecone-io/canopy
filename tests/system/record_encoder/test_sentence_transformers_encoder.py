import pytest

from canopy.knowledge_base.models import KBDocChunk
from canopy.knowledge_base.record_encoder.sentence_transformers import (
    SentenceTransformerRecordEncoder
)
from canopy.models.data_models import Query

documents = [KBDocChunk(
    id=f"doc_1_{i}",
    text=f"Sample document {i}",
    document_id=f"doc_{i}",
    metadata={"test": i},
    source="doc_1",
)
    for i in range(4)
]

queries = [Query(text="Sample query 1"),
           Query(text="Sample query 2"),
           Query(text="Sample query 3"),
           Query(text="Sample query 4")]


@pytest.fixture
def encoder():
    try:
        encoder = SentenceTransformerRecordEncoder(batch_size=2)
    except ImportError:
        pytest.skip(
            "`transformers` extra not installed. Skipping SentenceTransformer system "
            "tests"
        )
    return encoder


def test_dimension(encoder):
    assert encoder.dimension == 384


@pytest.mark.parametrize("items,function",
                         [(documents, "encode_documents"),
                          (queries, "encode_queries"),
                          ([], "encode_documents"),
                          ([], "encode_queries")])
def test_encode_documents(encoder, items, function):

    encoded_documents = getattr(encoder, function)(items)

    assert len(encoded_documents) == len(items)
    assert all(len(encoded.values) == encoder.dimension
               for encoded in encoded_documents)


@pytest.mark.asyncio
@pytest.mark.parametrize("items,function",
                         [("aencode_documents", documents),
                          ("aencode_queries", queries)])
async def test_aencode_not_implemented(encoder, function, items):
    with pytest.raises(NotImplementedError):
        await encoder.aencode_queries(items)
