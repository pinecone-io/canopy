import pytest

from canopy.knowledge_base.models import KBDocChunk
from canopy.knowledge_base.record_encoder.jina import JinaRecordEncoder
from canopy.models.data_models import Query

from unittest.mock import patch

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
    return JinaRecordEncoder(api_key='test_api_key', batch_size=2)


def test_dimension(encoder):
    with patch('pinecone_text.dense.JinaEncoder.encode_documents') \
            as mock_encode_documents:
        mock_encode_documents.return_value = [[0.1, 0.2, 0.3]]
        assert encoder.dimension == 3


def custom_encode(*args, **kwargs):
    input_to_encode = args[0]
    return [[0.1, 0.2, 0.3] for _ in input_to_encode]


@pytest.mark.parametrize("items,function",
                         [(documents, "encode_documents"),
                          (queries, "encode_queries"),
                          ([], "encode_documents"),
                          ([], "encode_queries")])
def test_encode_documents(encoder, items, function):
    with patch('pinecone_text.dense.JinaEncoder.encode_documents',
               side_effect=custom_encode):
        with patch('pinecone_text.dense.JinaEncoder.encode_queries',
                   side_effect=custom_encode):
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
