from typing import List

from canopy.chat_engine.query_generator import QueryGenerator
from canopy.models.data_models import Messages, Query, Role


class LastMessageQueryGenerator(QueryGenerator):
    """
        Returns the last message as a query without running any LLMs. This can be
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

        if len(messages) == 0:
            raise ValueError("Passed chat history does not contain any messages. "
                             "Please include at least one message in the history.")

        last_message = messages[-1]

        if last_message.role != Role.USER:
            raise ValueError(f"Expected a UserMessage, got {type(last_message)}.")

        return [Query(text=last_message.content)]

    async def agenerate(self,
                        messages: Messages,
                        max_prompt_tokens: int) -> List[Query]:
        return self.generate(messages, max_prompt_tokens)
