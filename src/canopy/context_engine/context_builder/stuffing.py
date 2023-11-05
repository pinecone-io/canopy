from itertools import zip_longest
from typing import List, Tuple

from canopy.context_engine.context_builder.base import ContextBuilder
from canopy.context_engine.models import (ContextQueryResult, ContextSnippet,
                                          StuffingContextContent, )
from canopy.knowledge_base.models import QueryResult, DocumentWithScore
from canopy.tokenizer import Tokenizer
from canopy.models.data_models import Context


class StuffingContextBuilder(ContextBuilder):

    def __init__(self):
        self._tokenizer = Tokenizer()

    def build(self,
              query_results: List[QueryResult],
              max_context_tokens: int) -> Context:

        sorted_docs_with_origin = self._round_robin_sort(query_results)

        # Stuff as many documents as possible into the context
        context_query_results = [
            ContextQueryResult(query=qr.query, snippets=[])
            for qr in query_results]
        debug_info = {"num_docs": len(sorted_docs_with_origin)}
        context = Context(
            content=StuffingContextContent(__root__=context_query_results),
            num_tokens=0,
            debug_info=debug_info
        )

        if self._tokenizer.token_count(context.to_text()) > max_context_tokens:
            return Context(content=StuffingContextContent(__root__=[]),
                           num_tokens=1, debug_info=debug_info)

        seen_doc_ids = set()
        for doc, origin_query_idx in sorted_docs_with_origin:
            if doc.id not in seen_doc_ids and doc.text.strip() != "":
                snippet = ContextSnippet(text=doc.text, source=doc.source)

                # try inserting the snippet into the context
                context_query_results[origin_query_idx].snippets.append(
                    snippet)
                seen_doc_ids.add(doc.id)
                # if the context is too long, remove the snippet
                if self._tokenizer.token_count(context.to_text()) > max_context_tokens:
                    context_query_results[origin_query_idx].snippets.pop()

        # remove queries with no snippets
        context.content = StuffingContextContent(
            __root__=[qr for qr in context_query_results if len(qr.snippets) > 0]
        )

        context.num_tokens = self._tokenizer.token_count(context.to_text())
        return context

    @staticmethod
    def _round_robin_sort(
            query_results: List[QueryResult]
    ) -> List[Tuple[DocumentWithScore, int]]:
        sorted_docs_with_origin = []

        for docs_tuple in zip_longest(*[qr.documents for qr in query_results],
                                      fillvalue=None):
            for idx, doc in enumerate(docs_tuple):
                if doc is not None:
                    sorted_docs_with_origin.append((doc, idx))

        return sorted_docs_with_origin

    async def abuild(self,
                     query_results: List[QueryResult],
                     max_context_tokens: int) -> Context:
        raise NotImplementedError()
