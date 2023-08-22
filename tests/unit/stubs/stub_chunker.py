from typing import List
from context_engine.knoweldge_base.chunker.base import Chunker
from context_engine.knoweldge_base.models import KBDocChunk
from context_engine.models.data_models import Document


class StubChunker(Chunker):
    def chunk_single_document(self, document: Document) -> List[KBDocChunk]:
        if document.text == "":
            return []
        return [KBDocChunk(id=f"{document.id}_0",
                           document_id=document.id,
                           text=document.text,
                           metadata=document.metadata)]

    async def achunk_single_document(self, document: Document) -> List[KBDocChunk]:
        raise NotImplementedError()
