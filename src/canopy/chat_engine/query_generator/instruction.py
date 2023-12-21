import logging
import re
from typing import List, Optional, cast

from tenacity import retry, stop_after_attempt, retry_if_exception_type

from canopy.chat_engine.history_pruner.raising import RaisingHistoryPruner
from canopy.chat_engine.query_generator import QueryGenerator, LastMessageQueryGenerator
from canopy.llm import BaseLLM, OpenAILLM
from canopy.models.api_models import ChatResponse
from canopy.models.data_models import Messages, Query, UserMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert on formulating a search query for a search engine,
to assist in responding to the user's question.

Given the following conversation, create a standalone question summarizing
 the user's last question, in its original language.

Reply to me in JSON in this format:

 {"question": {The question you generated here}}.

   Example:

        User: What is the weather today?

        Expected Response:
            ```json
            {"question": "What is the weather today?"}
            ```

    Example 2:

        User: How should I wash my white clothes in the laundry?
        Assistant: Separate from the colorful ones, and use a bleach.
        User: Which temperature?

        Expected Response:
         ```json
         {"question": "What is the right temperature for washing white clothes?"}
         ```


   Do not try to answer the question; just try to formulate a question representing the user's question.
   Do not return any other format other than the specified JSON format and keep it really short.

"""  # noqa: E501

USER_PROMPT = "Return only a JSON containing a single key 'question' and the value."


class ExtractionException(ValueError):
    pass


class InstructionQueryGenerator(QueryGenerator):
    _DEFAULT_COMPONENTS = {
        "llm": OpenAILLM,
    }

    def __init__(self,
                 *,
                 llm: Optional[BaseLLM] = None):
        """
             This `QueryGenerator` uses an LLM to formulate a knowledge base query
             from the full chat history. It does so by prompting the LLM to reply
             with a JSON containing a single key `question`, containing the query
             for the knowledge base. If LLM response cannot be parsed
             (after multiple retries), it falls back to returning the last message
             from the history as a query, much like `LastMessageQueryGenerator`
        """
        self._llm = llm or self._DEFAULT_COMPONENTS["llm"]()
        self._system_prompt = SYSTEM_PROMPT
        self._history_pruner = RaisingHistoryPruner()
        self._last_message_query_generator = LastMessageQueryGenerator()

        # Define a regex pattern to find the JSON object with the key "question"
        self._question_regex = re.compile(r'{\s*"question":\s*"([^"]+)"\s*}')

    def generate(self,
                 messages: Messages,
                 max_prompt_tokens: int) -> List[Query]:

        # Add a user message at the end; that helps us return a JSON object.
        new_history = (
                messages +
                [
                    UserMessage(content=USER_PROMPT)
                ]
        )

        chat_history = self._history_pruner.build(system_prompt=self._system_prompt,
                                                  chat_history=new_history,
                                                  max_tokens=max_prompt_tokens)

        question = self._try_generate_question(chat_history)

        if question is None:
            logger.warning("Falling back to the last message query generator.")
            return self._last_message_query_generator.generate(messages, 0)
        else:
            return [Query(text=question)]

    def _get_answer(self, messages: Messages) -> str:
        llm_response = self._llm.chat_completion(system_prompt=self._system_prompt,
                                                 chat_history=messages)
        response = cast(ChatResponse, llm_response)
        return response.choices[0].message.content

    @retry(stop=stop_after_attempt(3),
           retry=retry_if_exception_type(ExtractionException),
           retry_error_callback=lambda _: None)
    def _try_generate_question(self, messages: Messages) -> Optional[str]:
        content = self._get_answer(messages)
        return self._extract_question(content)

    def _extract_question(self, text: str) -> str:

        # Search for the pattern in the text
        match = re.search(self._question_regex, text)

        # If a match is found, extract and return the first occurrence
        if match:
            return match.group(1)

        raise ExtractionException("Failed to extract the question.")

    async def agenerate(self,
                        messages: Messages,
                        max_prompt_tokens: int) -> List[Query]:
        raise NotImplementedError
