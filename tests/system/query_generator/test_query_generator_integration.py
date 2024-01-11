import pytest

from canopy.tokenizer.tokenizer import Tokenizer # noqa
from canopy.llm.openai import OpenAILLM # noqa
from canopy.models.data_models import MessageBase, Query # noqa
from canopy.chat_engine.query_generator import FunctionCallingQueryGenerator # noqa
from typing import List # noqa


class TestFunctionCallingQueryGeneratorSystem:

    @staticmethod
    @pytest.fixture
    def openai_llm():
        Tokenizer.initialize()

    @staticmethod
    @pytest.fixture
    def query_generator(openai_llm):
        query_gen = FunctionCallingQueryGenerator(
            llm=openai_llm,
        )
        return query_gen

    @staticmethod
    @pytest.fixture
    def sample_messages():
        return [
            MessageBase(role="user", content="What is photosynthesis?")
        ]

    @staticmethod
    def test_generate_default_params(query_generator,
                                     sample_messages):
        result = query_generator.generate(messages=sample_messages,
                                          max_prompt_tokens=100)
        assert isinstance(result, List)
        assert len(result) > 0
        for query in result:
            assert isinstance(query, Query)
            assert len(query.text) > 0
