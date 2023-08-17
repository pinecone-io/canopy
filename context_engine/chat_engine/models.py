from enum import Enum


class HistoryPrunningMethod(Enum):
    RAISE = "raise",
    TRUNCATE = "truncate",
    SEARCH = "search"
