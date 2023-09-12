from typing import Optional, Tuple

from context_engine.chat_engine import ChatEngine
from context_engine.chat_engine.query_generator import QueryGenerator
from context_engine.context_engine import ContextEngine
from context_engine.context_engine.context_builder import ContextBuilder
from context_engine.knoweldge_base import KnowledgeBase
from context_engine.knoweldge_base.chunker import Chunker
from context_engine.knoweldge_base.record_encoder import RecordEncoder
from context_engine.knoweldge_base.reranker import Reranker
from context_engine.llm import BaseLLM

TOP_LEVEL_MANDATORY_KEYS = ["token_counting",
                            "knowledge_base",
                            "context_engine",
                            "llm"
                            "chat_engine"]

# def create_from_config(config: dict,
#                        *,
#                        index_name: str,
#                        llm_model_name: str,
#                        chunker: Optional[Chunker] = None,
#                        record_encoder: Optional[RecordEncoder] = None,
#                        reranker: Optional[Reranker] = None,
#                        context_builder: Optional[ContextBuilder] = None,
#                        llm: Optional[BaseLLM] = None,
#                        query_builder: Optional[QueryGenerator] = None,
#                        ) -> Tuple[KnowledgeBase, BaseLLM, ContextEngine, ChatEngine]:
#
#     kb_config = config["knowledge_base"]
#     kb = KnowledgeBase.create_from_config(kb_config,
#                                           index_name=index_name,
#                                           chunker=chunker,
#                                           record_encoder=record_encoder,
#                                           reranker=reranker)
#
#     context_engine_config = config["context_engine"]
#     context_engine = ContextEngine.create_from_config(context_engine_config,
#                                                       knowledge_base=kb,
#                                                       context_builder=context_builder)
#
#     if llm is None:
#         llm_config = config["llm"]
#         llm = BaseLLM.create_from_config(llm_config, model_name=llm_model_name)
#
#     chat_engine_config = config["chat_engine"]