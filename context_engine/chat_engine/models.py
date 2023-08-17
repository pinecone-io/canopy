from enum import Enum


class HistoryPruningMethod(Enum):
    RAISE = "raise",
    TRUNCATE = "truncate",
    SEARCH = "search"
