from typing import List

from pydantic import BaseModel

from resin.models.data_models import ContextContent


class ContextSnippet(BaseModel):
    source: str
    text: str


class ContextQueryResult(ContextContent):
    query: str
    snippets: List[ContextSnippet]

    def to_text(self):
        return self.json()
