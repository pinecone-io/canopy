from typing import List

from pydantic import BaseModel

from canopy.models.data_models import ContextContent


class ContextSnippet(BaseModel):
    source: str
    text: str


class ContextQueryResult(BaseModel):
    query: str
    snippets: List[ContextSnippet]


class StuffingContextContent(ContextContent):
    __root__: List[ContextQueryResult]

    def dict(self, **kwargs):
        return super().dict(**kwargs)['__root__']

    def to_text(self, **kwargs):
        return self.json(**kwargs)
