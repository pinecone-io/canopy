import openai


OPEN_AI_RETRY_EXCEPTIONS = (
    openai.error.Timeout,
    openai.error.APIConnectionError,
    openai.error.APIError,
    openai.error.RateLimitError,
    openai.error.ServiceUnavailableError
)
