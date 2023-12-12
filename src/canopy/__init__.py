import importlib.metadata
import warnings
import logging
import os
from typing import List

# Taken from https://stackoverflow.com/a/67097076
__version__ = importlib.metadata.version("canopy-sdk")


IGNORED_WARNINGS: List[str] = [
]

IGNORED_WARNING_IN_MODULES = [
    "transformers",
]

for warning in IGNORED_WARNINGS:
    warnings.filterwarnings("ignore", message=warning)
for module in IGNORED_WARNING_IN_MODULES:
    warnings.filterwarnings("ignore", module=module)
    logging.getLogger(module).setLevel(logging.ERROR)

# Apparently, `transformers` has its own logging system, and needs to be silenced separately # noqa: E501
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
