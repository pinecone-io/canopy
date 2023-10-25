from unittest.mock import create_autospec

import pytest

from canopy.llm.openai import OpenAILLM # noqa
from canopy.models.data_models import MessageBase, Query # noqa
from canopy.chat_engine.query_generator import FunctionCallingQueryGenerator # noqa
from canopy.chat_engine.prompt_builder import PromptBuilder # noqa
from typing import List # noqa


class TestFunctionCallingQueryGeneratorSystem:

    @staticmethod
    @pytest.fixture
    def openai_llm():
        return OpenAILLM(model_name="gpt-3.5-turbo")

    @staticmethod
    @pytest.fixture
    def prompt_builder():
        return create_autospec(PromptBuilder)

    @staticmethod
    @pytest.fixture
    def query_generator(openai_llm, prompt_builder):
        query_gen = FunctionCallingQueryGenerator(
            llm=openai_llm,
        )
        query_gen._prompt_builder = prompt_builder
        return query_gen

    @staticmethod
    @pytest.fixture
    def sample_messages():
        return [
            MessageBase(role="user", content="What is photosynthesis?")
        ]

    @staticmethod
    def test_generate_default_params(query_generator,
                                     prompt_builder,
                                     sample_messages):
        prompt_builder.build.return_value = sample_messages
        result = query_generator.generate(messages=sample_messages,
                                          max_prompt_tokens=100)
        assert isinstance(result, List)
        assert len(result) > 0
        for query in result:
            assert isinstance(query, Query)
            assert len(query.text) > 0
