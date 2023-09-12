from .base import RecordEncoder
from .dense_record_encoder import DenseRecordEncoder

ENCODER_CLASSES = {
    cls.__name__: cls for cls in RecordEncoder.__subclasses__()
}