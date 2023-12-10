from typing import List

from canopy.chat_engine.query_generator import QueryGenerator
from canopy.models.data_models import Messages, Query, Role


class LastMessageQueryGenerator(QueryGenerator):
    """
        Returns the last user message as a query without running any LLMs. This can be
        considered as the most basic query generation. Please use other query generators
        for more accurate results.
    """

    def generate(self,
                 messages: Messages,
                 max_prompt_tokens: int) -> List[Query]:
        """
            max_prompt_token is dismissed since we do not consume any token for
            generating the queries.
        """
        user_messages = [message for message in messages if message.role == Role.USER]

        if len(user_messages) == 0:
            raise ValueError("Passed chat history does not contain any user "
                             "messages. Please include at least one user message"
                             " in the history.")

        return [Query(text=user_messages[-1].content)]

    async def agenerate(self,
                        messages: Messages,
                        max_prompt_tokens: int) -> List[Query]:
        return self.generate(messages, max_prompt_tokens)
