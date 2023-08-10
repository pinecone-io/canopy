from typing import List

from context_engine.context_engine.context_builders.base_context_builder import BaseContextBuilder
from context_engine.context_engine.models import ContextQueryResult
from context_engine.knoweldge_base.models import KBQueryResult
from context_engine.knoweldge_base.tokenizers.base_tokenizer import Tokenizer
from context_engine.models.data_models import Context


class StuffingContextBuilder(BaseContextBuilder):

    def __init__(self,
                 tokenizer: Tokenizer,
                 reference_metadata_field: str,
                 tokens_safety_margin: int,
                 **kwargs):
        self._tokenizer = tokenizer
        self._reference_metadata_field = reference_metadata_field
        self._tokens_safety_margin = tokens_safety_margin

    def build_context(self,
                      query_results: List[KBQueryResult],
                      max_context_tokens: int,
    ) -> Context:

        # TODO - implement this method

        context_query_results: List[ContextQueryResult]
        actual_num_tokens: int
        debug_info: dict
        return Context(result=context_query_results,
                       num_tokens=actual_num_tokens,
                       debug_info=debug_info)


