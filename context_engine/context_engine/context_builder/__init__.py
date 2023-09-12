from ...utils import TypeDict as _TypeDict

from .stuffing import StuffingContextBuilder
from .base import ContextBuilder

CONTEXT_BUILDER_CLASSES: _TypeDict = {
    cls.__name__: cls for cls in ContextBuilder.__subclasses__()
}
CONTEXT_BUILDER_CLASSES['default'] = StuffingContextBuilder
