from unittest.mock import create_autospec

import pytest
import random

from canopy.chat_engine import ChatEngine
from canopy.chat_engine.query_generator import QueryGenerator
from canopy.context_engine import ContextEngine
from canopy.context_engine.context_builder.stuffing import (ContextSnippet,
                                                            ContextQueryResult,
                                                            StuffingContextContent, )
from canopy.llm import BaseLLM
from canopy.models.api_models import ChatResponse, _Choice, TokenCounts
from canopy.models.data_models import MessageBase, Query, Context, Role
from .. import random_words

MOCK_SYSTEM_PROMPT = "This is my mock prompt"
MAX_PROMPT_TOKENS = 100


class TestChatEngine:

    def setup_method(self):
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
            context_engine=self.mock_context_engine,
            llm=self.mock_llm,
            query_builder=self.mock_query_builder,
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
            content=StuffingContextContent(
                __root__=[ContextQueryResult(
                    query="How does photosynthesis work?",

                    snippets=[ContextSnippet(source="ref 1",
                                             text=self._generate_text(snippet_length)),
                              ContextSnippet(source="ref 2",
                                             text=self._generate_text(12))]
                )]
            ),
            num_tokens=1  # TODO: This is a dummy value. Need to improve.
        )

        mock_chat_response = ChatResponse(
            id='chatcmpl-7xuuGZzniUGiqxDSTJnqwb0l1xtfp',
            object='chat.completion',
            created=1694514456,
            model='gpt-3.5-turbo',
            choices=[_Choice(index=0,
                             message=MessageBase(
                                 role=Role.ASSISTANT,
                                 content="Photosynthesis is a process used by plants"),
                             finish_reason='stop')],
            usage=TokenCounts(prompt_tokens=25,
                              completion_tokens=9,
                              total_tokens=34),
            debug_info={})

        # Set the return values of the mocked methods
        self.mock_query_builder.generate.return_value = mock_queries
        self.mock_context_engine.query.return_value = mock_context
        self.mock_llm.chat_completion.return_value = mock_chat_response

        expected = {
            'queries': mock_queries,
            'prompt': system_prompt,
            'response': mock_chat_response,
            'context': mock_context,
        }
        return messages, expected

    def test_chat(self, namespace, history_length=5, snippet_length=10):

        chat_engine = self._init_chat_engine()

        # Mock input and expected output
        messages, expected = self._get_inputs_and_expected(history_length,
                                                           snippet_length,
                                                           MOCK_SYSTEM_PROMPT)

        # Call the method under test
        response = chat_engine.chat(messages, namespace=namespace)

        # Assertions
        self.mock_query_builder.generate.assert_called_once_with(
            messages,
            max_prompt_tokens=MAX_PROMPT_TOKENS
        )
        self.mock_context_engine.query.assert_called_once_with(
            expected['queries'],
            max_context_tokens=70,
            namespace=namespace
        )
        self.mock_llm.chat_completion.assert_called_once_with(
            system_prompt=expected['prompt'],
            context=expected['context'],
            chat_history=messages,
            stream=False,
            model_params={'max_tokens': 200}
        )

        assert response == expected['response']

    @pytest.mark.parametrize("allow_model_params_override,params_override",
                             [("False", None),
                              ("False", {'temperature': 0.99, 'top_p': 0.5}),
                              ("True", {'temperature': 0.99, 'top_p': 0.5}),
                              ("True", {'temperature': 0.99, 'max_tokens': 200}),],
                             ids=["no_override",
                                  "override_not_allowed",
                                  "valid_override",
                                  "valid_override_with_max_tokens"])
    def test_chat_engine_params(self,
                                namespace,
                                allow_model_params_override,
                                params_override,
                                system_prompt_length=10,
                                max_prompt_tokens=80,
                                max_context_tokens=60,
                                max_generated_tokens=150,
                                should_raise=False,
                                snippet_length=15,
                                history_length=3,
                                ):

        system_prompt = self._generate_text(system_prompt_length)
        chat_engine = self._init_chat_engine(
            system_prompt=system_prompt,
            max_prompt_tokens=max_prompt_tokens,
            max_context_tokens=max_context_tokens,
            max_generated_tokens=max_generated_tokens,
            allow_model_params_override=allow_model_params_override
        )

        # Mock input and expected output
        messages, expected = self._get_inputs_and_expected(history_length,
                                                           snippet_length,
                                                           system_prompt)

        # Call the method under test
        if should_raise:
            with pytest.raises(ValueError):
                chat_engine.chat(messages)
            return

        response = chat_engine.chat(messages,
                                    namespace=namespace,
                                    model_params=params_override)

        expected_model_params = {'max_tokens': max_generated_tokens}
        if allow_model_params_override and params_override is not None:
            expected_model_params.update(params_override)

        # Assertions
        self.mock_query_builder.generate.assert_called_once_with(
            messages,
            max_prompt_tokens=max_prompt_tokens
        )
        self.mock_context_engine.query.assert_called_once_with(
            expected['queries'],
            max_context_tokens=max_context_tokens,
            namespace=namespace
        )
        self.mock_llm.chat_completion.assert_called_once_with(
            system_prompt=expected['prompt'],
            context=expected['context'],
            chat_history=messages,
            stream=False,
            model_params=expected_model_params
        )

        assert response == expected['response']

    def test_context_tokens_too_small(self):
        system_prompt = self._generate_text(10)
        with pytest.raises(ValueError):
            self._init_chat_engine(system_prompt=system_prompt,
                                   max_prompt_tokens=15,
                                   max_context_tokens=10)

    def test_get_context(self):
        chat_engine = self._init_chat_engine()
        messages, expected = self._get_inputs_and_expected(5, 10, MOCK_SYSTEM_PROMPT)
        context = chat_engine._get_context(messages)

        self.mock_query_builder.generate.assert_called_once_with(
            messages,
            max_prompt_tokens=MAX_PROMPT_TOKENS
        )
        self.mock_context_engine.query.assert_called_once_with(
            expected['queries'],
            max_context_tokens=70,
            namespace=None
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
