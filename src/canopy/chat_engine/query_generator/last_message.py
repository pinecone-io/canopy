from typing import List

from canopy.chat_engine.query_generator import QueryGenerator
from canopy.models.data_models import Messages, Query


class LastMessageQueryGenerator(QueryGenerator):
    """
        Just returns the last message as a query without running any LLMs. This can be
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
            raise ValueError("Passed chat history does not contain any messages."
                             " Please include at least one message in the history.")
        return [Query(text=messages[-1].content)]

    async def agenerate(self,
                        messages: Messages,
                        max_prompt_tokens: int) -> List[Query]:
        return self.generate(messages, max_prompt_tokens)
