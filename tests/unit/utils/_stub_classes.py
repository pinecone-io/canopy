import abc
from typing import Optional

from canopy.utils.config import ConfigurableMixin


# A base class that inherits from ConfigurableMixin, with multiple derived classes
class BaseStubChunker(abc.ABC, ConfigurableMixin):
    @abc.abstractmethod
    def chunk(self, text: str) -> str:
        pass


class StubChunker(BaseStubChunker):
    DEFAULT_CHUNK_SIZE = 100
    DEFAULT_SPLITTER = ' '

    def __init__(self, chunk_size=DEFAULT_CHUNK_SIZE, splitter=DEFAULT_SPLITTER):
        self.chunk_size = chunk_size
        self.splitter = splitter

    def chunk(self, text: str) -> str:
        return text.split(self.splitter)


class StubOtherChunker(BaseStubChunker):
    DEFAULT_CHUNK_SIZE = 200

    def __init__(self, chunk_size=DEFAULT_CHUNK_SIZE, some_param=' '):
        self.chunk_size = chunk_size
        self.splitter = some_param

    def chunk(self, text: str) -> str:
        return text.split(self.splitter)


# A base class that inherits from ConfigurableMixin, where the derived class has
# default components
class BaseStubKB(abc.ABC, ConfigurableMixin):
    pass


class StubKB(BaseStubKB):
    _DEFAULT_COMPONENTS = {
        'chunker': StubChunker,
    }

    DEFAULT_TOP_K = 5

    def __init__(self,
                 chunker: Optional[BaseStubChunker] = None,
                 top_k: int = DEFAULT_TOP_K,
                 ):
        self.chunker = chunker or self._DEFAULT_COMPONENTS['chunker']()
        self.top_k = top_k


class BaseStubContextBuilder(ConfigurableMixin):
    pass


class StubContextBuilder(BaseStubContextBuilder):
    DEFAULT_MAX_CONTEXT_LENGTH = 1000

    def __init__(self, max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH):
        self.max_context_length = max_context_length


# A base class that inherits from ConfigurableMixin, where the derived class has
# default components, one of them is a class that also inherits from ConfigurableMixin
class BaseStubContextEngine(ConfigurableMixin):
    pass


class StubContextEngine(BaseStubContextEngine):
    _DEFAULT_COMPONENTS = {
        'knowledge_base': StubKB,
        'context_builder': StubContextBuilder,
    }

    def __init__(self,
                 knowledge_base: StubKB,
                 context_builder: Optional[BaseStubContextBuilder] = None,
                 filter: Optional[dict] = None,
                 ):
        self.knowledge_base = knowledge_base or self._DEFAULT_COMPONENTS['kb']()
        self.context_builder = (context_builder or
                                self._DEFAULT_COMPONENTS['context_builder']())
        self.filter = filter
