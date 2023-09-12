from typing import Optional, Tuple

from context_engine.chat_engine import ChatEngine
from context_engine.chat_engine.query_generator import QueryGenerator
from context_engine.context_engine import ContextEngine
from context_engine.context_engine.context_builder import ContextBuilder
from context_engine.knoweldge_base import KnowledgeBase
from context_engine.knoweldge_base.chunker import Chunker
from context_engine.knoweldge_base.record_encoder import RecordEncoder
from context_engine.knoweldge_base.reranker import Reranker
from context_engine.llm import BaseLLM, LLM_CLASSES
from context_engine.utils import initialize_from_config

MANDATORY_KEYS = ["knowledge_base",
                  "context_engine"]

ALLOWED_KEYS = MANDATORY_KEYS + ["tokenizer", "llm", "chat_engine"]


def create_from_config(config: dict,
                       *,
                       index_name: str,
                       chunker: Optional[Chunker] = None,
                       record_encoder: Optional[RecordEncoder] = None,
                       reranker: Optional[Reranker] = None,
                       context_builder: Optional[ContextBuilder] = None,
                       llm: Optional[BaseLLM] = None,
                       query_builder: Optional[QueryGenerator] = None,
                       ) -> Tuple[KnowledgeBase,
                                  ContextEngine,
                                  Optional[BaseLLM],
                                  Optional[ChatEngine]]:
    """

    Args:
        config:
        index_name:
        chunker:
        record_encoder:
        reranker:
        context_builder:
        llm:
        query_builder:

    Returns:

    """
    # Check mandatory keys
    config_keys = set(config.keys())
    missing_keys = set(MANDATORY_KEYS) - config_keys
    if missing_keys:
        raise ValueError(f"Missing mandatory keys in config: {list(missing_keys)}")

    # Check unallowed keys
    unallowed_keys = config_keys - set(ALLOWED_KEYS)
    if unallowed_keys:
        raise ValueError(f"Unallowed keys in config: {list(unallowed_keys)}")

    # Tokenizer
    # TODO: add tokenizer to config + add Tokenizer default class

    # Knowledge base
    kb_config = config["knowledge_base"]
    kb = KnowledgeBase.from_config(kb_config,
                                   index_name=index_name,
                                   chunker=chunker,
                                   record_encoder=record_encoder,
                                   reranker=reranker)

    # Context engine
    context_engine_config = config["context_engine"]
    context_engine = ContextEngine.from_config(context_engine_config,
                                               knowledge_base=kb,
                                               context_builder=context_builder)

    # LLM
    override_err_msg = "Cannot provide both {key} override and {key} config. " \
                       "If you wish to override with your own {key}, " \
                       "remove the '{key}' key from the config"

    llm_config = config.get("llm", None)
    if llm and llm_config:
        raise ValueError(override_err_msg.format(key="llm"))
    if llm_config:
        llm = initialize_from_config(llm_config, LLM_CLASSES, "llm")

    # Chat engine
    chat_engine_config = config.get("chat_engine", None)
    if chat_engine_config:
        if llm is None:
            raise ValueError(
                "Cannot initialize ChatEngine without an LLM. Please provide "
                "an LLM or add an 'llm' section to the config")
        chat_engine = ChatEngine.from_config(chat_engine_config,
                                             llm=llm,
                                             context_engine=context_engine,
                                             query_builder=query_builder)
    else:
        chat_engine = None

    return kb, context_engine, llm, chat_engine
