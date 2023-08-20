from typing import List
from context_engine.knoweldge_base.chunker.base import Chunker
from context_engine.knoweldge_base.models import KBDocChunk
from context_engine.models.data_models import Document


class StubChunker(Chunker):
    def chunk_single_document(self, document: Document) -> List[KBDocChunk]:
        tokens = document.text.split()
        return [KBDocChunk(id=f"{document.id}_{i}",
                           document_id=document.id,
                           text=tokens[i],
                           metadata=document.metadata)
                for i in range(len(tokens))]

    async def achunk_single_document(self, document: Document) -> List[KBDocChunk]:
        raise NotImplementedError()
