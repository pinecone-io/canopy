from typing import List, Optional

from pydantic import BaseModel

from context_engine.models.data_models import ContextContent


class ContextSnippet(BaseModel):
    reference: Optional[str]
    text: str


class ContextQueryResult(ContextContent):
    query: str
    snippets: List[ContextSnippet]

    def to_text(self):
        return self.json()
