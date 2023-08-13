from typing import List

from context_engine.context_engine.context_builder.base import BaseContextBuilder
from context_engine.context_engine.models import ContextQueryResult
from context_engine.knoweldge_base.models import QueryResult
from context_engine.knoweldge_base.tokenizer.base import Tokenizer
from context_engine.models.data_models import Context


class StuffingContextBuilder(BaseContextBuilder):

    def __init__(self,
                 tokenizer: Tokenizer,
                 reference_metadata_field: str,
                 tokens_safety_margin: int,
                 ):
        self._tokenizer = tokenizer
        self._reference_metadata_field = reference_metadata_field
        self._tokens_safety_margin = tokens_safety_margin

    def build(self, query_results: List[QueryResult], max_context_tokens: int, ) -> Context:
        # TODO - implement this method

        context_query_results: List[ContextQueryResult]
        actual_num_tokens: int
        debug_info: dict
        return Context(result=context_query_results, num_tokens=actual_num_tokens, debug_info=debug_info)
