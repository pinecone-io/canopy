from ...utils import TypeDict as _TypeDict


from .base import QueryGenerator
from .function_calling import FunctionCallingQueryGenerator

QUERY_GENERATOR_CLASSES: _TypeDict = {
    cls.__name__: cls for cls in QueryGenerator.__subclasses__()
}
QUERY_GENERATOR_CLASSES['default'] = FunctionCallingQueryGenerator