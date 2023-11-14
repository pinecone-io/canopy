import openai

from importlib.metadata import version

from packaging.version import parse

def is_openai_v1() -> bool:
    _version = parse(version("openai"))
    return _version.major >= 1

OPEN_AI_TRANSIENT_EXCEPTIONS = (
    openai.error.Timeout,
    openai.error.APIConnectionError,
    openai.error.APIError,
    openai.error.RateLimitError,
    openai.error.ServiceUnavailableError
)
