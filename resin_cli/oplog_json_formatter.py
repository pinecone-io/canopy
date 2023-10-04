import json

from oplog import Operation
from oplog.formatters import BaseOperationFormatter


class OplogJsonFormatter(BaseOperationFormatter):
    def format_op(self, op: Operation) -> str:
        optional_fields = {
            "exception_type": str(op.exception_type or ""),
            "traceback": str(op.traceback or ""),
            "custom_props": op.custom_props,
            "global_props": op.global_props,
        }

        row = {
            "start_time_utc": op.start_time_utc_str,
            "name": op.name,
            "result": op.result,
            "duration_ms": str(op.duration_ms),
            "correlation_id": op.correlation_id,
            **{k: v for k, v in optional_fields.items() if v},
        }

        return json.dumps(row, ensure_ascii=False)
