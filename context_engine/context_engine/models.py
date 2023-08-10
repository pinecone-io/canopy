from typing import List
from pydantic import BaseModel


class ContextSnippet(BaseModel):
    reference: str
    text: str


class ContextQueryResult(BaseModel):
    query: str
    snippets: List[ContextSnippet]