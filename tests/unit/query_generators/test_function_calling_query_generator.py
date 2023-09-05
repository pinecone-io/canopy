from typing import List
from unittest.mock import create_autospec

import pytest

from context_engine.chat_engine.prompt_builder import PromptBuilder
from context_engine.chat_engine.query_generator.function_calling \
    import (FunctionCallingQueryGenerator, DEFAULT_FUNCTION_DESCRIPTION,
            DEFAULT_SYSTEM_PROMPT, )
from context_engine.llm import BaseLLM
from context_engine.llm.models import (ModelParams, Function,
                                       FunctionParameters, FunctionArrayProperty, )
from context_engine.models.data_models import Query, MessageBase
from stubs.stub_tokenizer import StubTokenizer


class TestFunctionCallingQueryGenerator:

    @staticmethod
    @pytest.fixture
    def mock_llm():
        return create_autospec(BaseLLM)

    @staticmethod
    @pytest.fixture
    def mock_prompt_builder():
        return create_autospec(PromptBuilder)

    @staticmethod
    @pytest.fixture
    def mock_model_params():
        return create_autospec(ModelParams)

    @staticmethod
    @pytest.fixture
    def query_generator(mock_llm, mock_prompt_builder, mock_model_params):
        query_gen = FunctionCallingQueryGenerator(
            llm=mock_llm,
            top_k=5,
            tokenizer=StubTokenizer()
        )
        query_gen._prompt_builder = mock_prompt_builder
        return query_gen

    @staticmethod
    @pytest.fixture
    def sample_messages():
        return [
            MessageBase(role="user", content="What is photosynthesis?")
        ]

    @staticmethod
    def test_generate_with_default_params(query_generator,
                                          mock_llm,
                                          mock_prompt_builder,
                                          sample_messages
                                          ):
        mock_prompt_builder.build.return_value = sample_messages
        mock_llm.enforced_function_call.return_value = {"queries": ["query1", "query2"]}

        result = query_generator.generate(messages=sample_messages,
                                          max_prompt_tokens=100)

        mock_prompt_builder.build.assert_called_once_with(
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            history=sample_messages,
            max_tokens=100
        )

        expected_function = Function(
            name="query_knowledgebase",
            description=DEFAULT_FUNCTION_DESCRIPTION,
            parameters=FunctionParameters(
                required_properties=[
                    FunctionArrayProperty(
                        name="queries",
                        items_type="string",
                        description='List of queries to send to the search engine.',
                    ),
                ]
            ),
        )
        assert mock_llm.enforced_function_call.called
        args, kwargs = mock_llm.enforced_function_call.call_args
        assert kwargs['function'] == expected_function

        # Ensure the result is correct
        assert isinstance(result, List)
        assert len(result) == 2
        assert result[0] == Query(text="query1", top_k=5)
        assert result[1] == Query(text="query2", top_k=5)

    @staticmethod
    def test_generate_with_non_defaults(query_generator,
                                        mock_llm,
                                        mock_prompt_builder,
                                        sample_messages
                                        ):
        custom_system_prompt = "Custom system prompt"
        custom_function_description = "Custom function description"

        gen_custom = FunctionCallingQueryGenerator(
            llm=mock_llm,
            top_k=5,
            prompt=custom_system_prompt,
            function_description=custom_function_description,
            tokenizer=StubTokenizer()
        )
        gen_custom._prompt_builder = mock_prompt_builder

        mock_prompt_builder.build.return_value = sample_messages
        mock_llm.enforced_function_call.return_value = {"queries": ["query1"]}

        result = gen_custom.generate(messages=sample_messages,
                                     max_prompt_tokens=100)

        expected_result = [Query(text="query1", top_k=5)]
        assert result == expected_result

        mock_prompt_builder.build.assert_called_once_with(
            system_prompt=custom_system_prompt,
            history=sample_messages,
            max_tokens=100
        )

        expected_function = Function(
            name="query_knowledgebase",
            description=custom_function_description,
            parameters=FunctionParameters(
                required_properties=[
                    FunctionArrayProperty(
                        name="queries",
                        items_type="string",
                        description='List of queries to send to the search engine.',
                    ),
                ]
            ),
        )

        assert mock_llm.enforced_function_call.called
        args, kwargs = mock_llm.enforced_function_call.call_args
        assert kwargs['function'] == expected_function

    @staticmethod
    def test_generate_invalid_return_from_llm(query_generator,
                                              mock_llm,
                                              mock_prompt_builder,
                                              sample_messages
                                              ):
        mock_prompt_builder.build.return_value = sample_messages
        mock_llm.enforced_function_call.return_value = {}

        with pytest.raises(KeyError):
            query_generator.generate(messages=sample_messages,
                                     max_prompt_tokens=100)

    @staticmethod
    @pytest.mark.asyncio
    async def test_agenerate_not_implemented(query_generator,
                                             mock_llm,
                                             mock_prompt_builder,
                                             sample_messages
                                             ):
        with pytest.raises(NotImplementedError):
            await query_generator.agenerate(messages=sample_messages,
                                            max_prompt_tokens=100)
