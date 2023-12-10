from typing import List

from canopy.chat_engine.query_generator import QueryGenerator
from canopy.models.data_models import Messages, Query


class LastMessageQueryGenerator(QueryGenerator):

    def generate(self,
                 messages: Messages,
                 max_prompt_tokens: int) -> List[Query]:
        return [Query(text=messages[-1].content)]

    async def agenerate(self,
                        messages: Messages,
                        max_prompt_tokens: int) -> List[Query]:
        return self.generate(messages, max_prompt_tokens)
