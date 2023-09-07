from enum import Enum


class HistoryPruningMethod(Enum):
    RAISE = "raise"
    RECENT = "recent"
