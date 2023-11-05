from typing import List

from pydantic import BaseModel

from canopy.models.data_models import _ContextContent


class ContextSnippet(BaseModel):
    source: str
    text: str


class ContextQueryResult(_ContextContent):
    query: str
    snippets: List[ContextSnippet]

    def to_text(self, **kwargs):
        return self.json(**kwargs)
