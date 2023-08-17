from itertools import zip_longest
from typing import List, Tuple

from context_engine.context_engine.context_builder.base import BaseContextBuilder
from context_engine.context_engine.models import ContextQueryResult, ContextSnippet
from context_engine.knoweldge_base.models import QueryResult, DocumentWithScore
from context_engine.knoweldge_base.tokenizer.base import Tokenizer
from context_engine.models.data_models import Context


class StuffingContextBuilder(BaseContextBuilder):

    def __init__(self,
                 tokenizer: Tokenizer,
                 source_metadata_field: str,
                 include_queries: bool = True,):
        self._tokenizer = tokenizer
        self._source_metadata_field = source_metadata_field
        self._include_queries = include_queries

    def build(self, query_results: List[QueryResult], max_context_tokens: int) -> Context:
        # Initial variables
        num_tokens = 0
        sorted_docs_with_origin = self._round_robin_sort(query_results)

        # Stuff as many documents as possible into the context
        context_query_results: List[ContextQueryResult] = [ContextQueryResult(query=qr.query, snippets=[])
                                                           for qr in query_results]
        debug_info = {"num_docs": len(sorted_docs_with_origin)}
        context = Context(content=context_query_results, num_tokens=num_tokens, debug_info=debug_info)
        inserted_doc_ids = set()
        for doc, origin_query_idx in sorted_docs_with_origin:
            doc_tokens = self._tokenizer.token_count(doc.text)
            if (num_tokens + doc_tokens <= max_context_tokens) and doc.id not in inserted_doc_ids:
                snippet = ContextSnippet(text=doc.text, source=doc.metadata.get(self._source_metadata_field, None))
                context_query_results[origin_query_idx].snippets.append(snippet)
                num_tokens += doc_tokens
                inserted_doc_ids.add(doc.id)
        return context

    @staticmethod
    def _round_robin_sort(query_results: List[QueryResult]) -> List[Tuple[DocumentWithScore, int]]:
        """
        Sort documents from query results in a round-robin manner.

        Args:
            query_results (List[QueryResult]): The list of query results containing documents.

        Returns:
            List[Tuple[DocumentWithScore, int]]: List of tuples where each tuple contains a document
                                                 and its origin index.
        """
        sorted_docs_with_origin = []

        for docs_tuple in zip_longest(*[qr.documents for qr in query_results], fillvalue=None):
            for idx, doc in enumerate(docs_tuple):
                if doc is not None:
                    sorted_docs_with_origin.append((doc, idx))

        return sorted_docs_with_origin

    async def abuild(self, query_results: List[QueryResult], max_context_tokens: int) -> Context:
        raise NotImplementedError()

