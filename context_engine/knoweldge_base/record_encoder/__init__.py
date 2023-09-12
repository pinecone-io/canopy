from .base import RecordEncoder
from .dense_record_encoder import DenseRecordEncoder
from ...utils import TypeDict as _TypeDict


ENCODER_CLASSES: _TypeDict = {
    cls.__name__: cls for cls in RecordEncoder.__subclasses__()
}
ENCODER_CLASSES['default'] = DenseRecordEncoder
