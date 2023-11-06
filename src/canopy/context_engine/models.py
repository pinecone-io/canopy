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

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]

    def __len__(self):
        return len(self.__root__)

    def to_text(self, **kwargs):
        return self.json(**kwargs)
