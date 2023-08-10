from typing import List, Optional

from pinecone_text.sparse import SparseVector
from pydantic import BaseModel, Field

from context_engine.models.data_models import Document, Query

# TODO 1: consider moving this to pinecone-text
# TODO 2: consider renaming to "Vector" or "DenseVector"
# TODO 3: consider supporting `np.ndarray`
VectorValues = List[float]


class KBDocChunk(Document):
    values: Optional[VectorValues] = None
    sparse_values: Optional[SparseVector] = None


class KBDocChunkWithScore(KBDocChunk):
    score: float


class KBQuery(Query):
    values: Optional[VectorValues] = None
    sparse_values: Optional[SparseVector] = None


class KBQueryResult(BaseModel):
    query: str
    documents: List[KBDocChunkWithScore]


class DocumentWithScore(Document):
    score: float


class QueryResult(BaseModel):
    query: str
    documents: List[DocumentWithScore]
    debug_info: dict = Field(default_factory=dict, exclude=True)
