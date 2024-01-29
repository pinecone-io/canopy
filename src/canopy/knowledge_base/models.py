from copy import deepcopy
from typing import List, Optional

from pydantic import BaseModel, Field

from canopy.models.data_models import Document, Query, SparseVector

# TODO: (1) consider moving this to pinecone-text
# TODO: (2) consider renaming to "Vector" or "DenseVector"
# TODO: (3) consider supporting `np.ndarray`
VectorValues = List[float]


class KBDocChunk(Document):
    document_id: str


class KBDocChunkWithScore(KBDocChunk):
    score: float


class KBEncodedDocChunk(KBDocChunk):
    values: VectorValues
    sparse_values: Optional[SparseVector] = None

    def to_db_record(self):
        metadata = deepcopy(self.metadata)
        metadata["text"] = self.text
        metadata["document_id"] = self.document_id
        metadata["source"] = self.source

        record = {
            "id": self.id,
            "values": self.values,
            "metadata": metadata,

        }

        if self.sparse_values is not None and len(self.sparse_values["values"]) > 0:
            record["sparse_values"] = self.sparse_values

        return record


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
