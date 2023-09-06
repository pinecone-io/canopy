from unittest.mock import create_autospec

import pytest
import random

from context_engine.chat_engine import ChatEngine
from context_engine.chat_engine.prompt_builder.prompt_builder import BasePromptBuilder
from context_engine.chat_engine.query_generator import QueryGenerator
from context_engine.context_engine import ContextEngine
from context_engine.context_engine.models import ContextQueryResult, ContextSnippet
from context_engine.llm import BaseLLM
from context_engine.llm.models import UserMessage, SystemMessage
from context_engine.models.data_models import MessageBase, Role, Query, Context
from ..stubs.stub_tokenizer import StubTokenizer
from .. import random_words

MOCK_SYSTEM_PROMPT = "This is my mock prompt"
MAX_PROMPT_TOKENS = 100


class TestChatEngine:

    # @classmethod
    # def setup_class(cls):
    #     # cls.mock_prompt_builder = create_autospec(BasePromptBuilder)
    #     cls.mock_llm = create_autospec(BaseLLM)
    #     cls.mock_query_builder = create_autospec(QueryGenerator)
    #     cls.mock_context_engine = create_autospec(ContextEngine)

    def setup(self):
        # self.mock_prompt_builder = create_autospec(BasePromptBuilder)
        self.mock_llm = create_autospec(BaseLLM)
        self.mock_query_builder = create_autospec(QueryGenerator)
        self.mock_context_engine = create_autospec(ContextEngine)

    def _init_chat_engine(self,
                          system_prompt: str = MOCK_SYSTEM_PROMPT,
                          max_prompt_tokens: int = MAX_PROMPT_TOKENS,
                          max_context_tokens: int = None,
                          max_generated_tokens: int = 200,
                          **kwargs):
        return ChatEngine(
            llm=self.mock_llm,
            context_engine=self.mock_context_engine,
            query_builder=self.mock_query_builder,
            tokenizer=StubTokenizer(),
            system_prompt=system_prompt,
            max_prompt_tokens=max_prompt_tokens,
            max_context_tokens=max_context_tokens,
            max_generated_tokens=max_generated_tokens,
            **kwargs
        )

    @staticmethod
    def _generate_text(num_words: int):
        return " ".join(random.choices(random_words, k=num_words))

    def _get_inputs_and_expected(self,
                                 history_length,
                                 snippet_length,
                                 system_prompt):
        messages = [
            MessageBase(
                role="assistant" if i % 2 == 0 else "user",
                content=self._generate_text(5)
            )
            for i in range(history_length)
        ]
        mock_queries = [Query(text="How does photosynthesis work?")]
        mock_context = Context(
            content=ContextQueryResult(
                query="How does photosynthesis work?",
                snippets=[ContextSnippet(reference="ref 1",
                                         text=self._generate_text(snippet_length)),
                          ContextSnippet(reference="ref 2",
                                         text=self._generate_text(12))]
            ),
            num_tokens=1  # TODO: This is a dummy value. Need to improve.
        )
        expected_prompt = [SystemMessage(
            content=system_prompt + f"\nContext: {mock_context.to_text()}"
        )] + messages
        mock_chat_response = "Photosynthesis is a process used by plants..."

        # Set the return values of the mocked methods
        self.mock_query_builder.generate.return_value = mock_queries
        self.mock_context_engine.query.return_value = mock_context
        self.mock_llm.chat_completion.return_value = mock_chat_response

        expected = {
            'queries': mock_queries,
            'prompt': expected_prompt,
            'response': mock_chat_response,
            'context': mock_context,
        }
        return messages, expected

    def test_chat(self, history_length=5, snippet_length=10):
        chat_engine = self._init_chat_engine()

        # Mock input and expected output
        messages, expected = self._get_inputs_and_expected(history_length,
                                                           snippet_length,
                                                           MOCK_SYSTEM_PROMPT)

        # Call the method under test
        response = chat_engine.chat(messages)

        # Assertions
        self.mock_query_builder.generate.assert_called_once_with(
            messages,
            max_prompt_tokens=MAX_PROMPT_TOKENS
        )
        self.mock_context_engine.query.assert_called_once_with(
            expected['queries'],
            max_context_tokens=70
        )
        self.mock_llm.chat_completion.assert_called_once_with(
            expected['prompt'],
            max_tokens=200,
            stream=False,
            model_params=None
        )

        assert response == expected['response']

    # TODO: parametrize and add more test cases
    def test_chat_engine_params(self,
                                system_prompt_length=10,
                                max_prompt_tokens=80,
                                max_context_tokens=60,
                                max_generated_tokens=150,
                                should_raise=False,
                                snippet_length=15,
                                history_length=3,
                                ):

        system_prompt = self._generate_text(system_prompt_length)
        chat_engine = self._init_chat_engine(system_prompt=system_prompt,
                                             max_prompt_tokens=max_prompt_tokens,
                                             max_context_tokens=max_context_tokens,
                                             max_generated_tokens=max_generated_tokens)

        # Mock input and expected output
        messages, expected = self._get_inputs_and_expected(history_length,
                                                           snippet_length,
                                                           system_prompt)

        # Call the method under test
        if should_raise:
            with pytest.raises(ValueError):
                chat_engine.chat(messages)
            return

        response = chat_engine.chat(messages)

        # Assertions
        self.mock_query_builder.generate.assert_called_once_with(
            messages,
            max_prompt_tokens=max_prompt_tokens
        )
        self.mock_context_engine.query.assert_called_once_with(
            expected['queries'],
            max_context_tokens=max_context_tokens
        )
        self.mock_llm.chat_completion.assert_called_once_with(
            expected['prompt'],
            max_tokens=max_generated_tokens,
            stream=False,
            model_params=None
        )

        assert response == expected['response']

    def test_context_tokens_to_small(self):
        system_prompt = self._generate_text(10)
        with pytest.raises(ValueError):
            self._init_chat_engine(system_prompt=system_prompt, max_context_tokens=10)

    def test_prompt_tokens_to_small(self):
        system_prompt = self._generate_text(10)
        with pytest.raises(ValueError):
            self._init_chat_engine(system_prompt=system_prompt, max_prompt_tokens=10)

    def test_get_context(self):
        chat_engine = self._init_chat_engine()
        messages, expected = self._get_inputs_and_expected(5, 10, MOCK_SYSTEM_PROMPT)
        context = chat_engine.get_context(messages)

        self.mock_query_builder.generate.assert_called_once_with(
            messages,
            max_prompt_tokens=MAX_PROMPT_TOKENS
        )
        self.mock_context_engine.query.assert_called_once_with(
            expected['queries'],
            max_context_tokens=70
        )

        assert isinstance(context, Context)
        assert context == expected['context']

    @pytest.mark.asyncio
    async def test_aget_context_raise(self):
        chat_engine = self._init_chat_engine()
        with pytest.raises(NotImplementedError):
            await chat_engine.aget_context([])

    @pytest.mark.asyncio
    async def test_achat_raise(self):
        chat_engine = self._init_chat_engine()
        with pytest.raises(NotImplementedError):
            await chat_engine.achat([])
