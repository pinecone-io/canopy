from context_engine.chat_engine.query_builder.base import QueryBuilder

from typing import Optional

from context_engine.chat_engine.models import HistoryPruningMethod
from context_engine.llm.base import BaseLLM

DEFAULT_TEMPLATE = """When you receive a user question regarding {topic} {topic_description}, your task is to formulate one or more search queries to retrieve relevant information from a search engine. 
You should break down complex questions into sub-queries if needed."""  # noqa


class FunctionCallingQueryBuilder(QueryBuilder):
    def __init__(self,
                 llm: BaseLLM,
                 *,
                 topic: str,
                 topic_description: str,
                 prompt: Optional[str] = None,
                 prompt_template: Optional[str] = None,
                 history_pruning: HistoryPruningMethod = HistoryPruningMethod.RAISE):
        self.llm = llm

        if prompt is not None and prompt_template is not None:
            raise ValueError("`prompt` and `prompt_template` cannot be set together")
        if prompt:
            self.prompt = prompt
        elif prompt_template:
            self.prompt = prompt_template.format(topic=topic,
                                                 topic_description=topic_description)
        else:
            self.prompt = DEFAULT_TEMPLATE.format(topic=topic,
                                                  topic_description=topic_description)
        self.history_pruning = history_pruning
