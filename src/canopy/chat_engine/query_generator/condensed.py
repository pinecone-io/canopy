import logging
import re
from typing import List, Optional

from tenacity import retry, stop_after_attempt, retry_if_exception_type

from canopy.chat_engine.models import HistoryPruningMethod
from canopy.chat_engine.prompt_builder import PromptBuilder
from canopy.chat_engine.query_generator import QueryGenerator, LastMessageQueryGenerator
from canopy.llm import BaseLLM, AnyscaleLLM
from canopy.models.data_models import Messages, Query, UserMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert on formulating a search query for a search engine, to assist
 in responding to the user's question.

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

USER_PROMPT = "Return only a JSON that has 'question' as a key and the value."


class ExtractionException(ValueError):
    pass


class CondensedQueryGenerator(QueryGenerator):
    _DEFAULT_COMPONENTS = {
        "llm": AnyscaleLLM,
    }

    def __init__(self,
                 *,
                 llm: Optional[BaseLLM] = None):
        """
             Tries to build a question out of all the chat history.
             In order to do that, the system first tries to make LLM return
             a JSON object with a key "question" and the value representing
             the question the user was intending to ask. If LLM response
             cannot be parsed it falls back to last message query generator,
             meaning it returns the last message of the history as a query.
        """
        self._llm = llm or self._DEFAULT_COMPONENTS["llm"]()
        self._system_prompt = SYSTEM_PROMPT
        self._prompt_builder = PromptBuilder(HistoryPruningMethod.RAISE, 2)
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

        new_messages = self._prompt_builder.build(system_prompt=self._system_prompt,
                                                  history=new_history,
                                                  max_tokens=max_prompt_tokens)

        question = self._try_generate_question(new_messages)

        if question is None:
            logger.warning("Falling back to the last message query generator.")
            return self._last_message_query_generator.generate(messages, 0)
        else:
            return [Query(text=question)]

    def _get_answer(self, messages: Messages) -> str:
        response = self._llm.chat_completion(messages)
        return response.choices[0].message.content

    @retry(stop=stop_after_attempt(3),
           retry=retry_if_exception_type(ExtractionException),
           retry_error_callback=lambda _: None)
    def _try_generate_question(self, messages: Messages) -> Optional[str]:
        content = self._get_answer(messages)
        question = self._try_extract_question(content)

        if question is None:
            raise ExtractionException("Failed to extract the question.")
        else:
            return question

    def _try_extract_question(self, text: str) -> Optional[str]:

        # Search for the pattern in the text
        match = re.search(self._question_regex, text)

        # If a match is found, extract and return the first occurrence
        if match:
            return match.group(1)

        return None

    async def agenerate(self,
                        messages: Messages,
                        max_prompt_tokens: int) -> List[Query]:
        raise NotImplementedError
