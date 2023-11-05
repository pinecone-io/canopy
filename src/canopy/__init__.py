import importlib.metadata

from .tokenizer import Tokenizer
from .knowledge_base import KnowledgeBase
from .context_engine import ContextEngine
from .chat_engine import ChatEngine

from .models.data_models import (
    Document, Query, QueryResult, Context, ContextContent,
    Messages, UserMessage, SystemMessage, AssistantMessage,
)


# Taken from https://stackoverflow.com/a/67097076
__version__ = importlib.metadata.version("pinecone-canopy")
