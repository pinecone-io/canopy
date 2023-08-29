import pytest
from unittest.mock import create_autospec
from context_engine.chat_engine.base import ChatEngine
from context_engine.chat_engine.prompt_builder.base import BasePromptBuilder
from context_engine.chat_engine.query_generator.base import QueryGenerator
from context_engine.knoweldge_base.base_knoweldge_base import BaseKnowledgeBase
from context_engine.knoweldge_base.models import QueryResult
from context_engine.llm.base import BaseLLM
from context_engine.llm.models import ModelParams
from context_engine.models.data_models import MessageBase, Role, Query


class TestChatEngine:

    @staticmethod
    @pytest.fixture
    def mock_llm():
        return create_autospec(BaseLLM)

    @staticmethod
    @pytest.fixture
    def mock_query_builder():
        return create_autospec(QueryGenerator)

    @staticmethod
    @pytest.fixture
    def mock_knowledge_base():
        return create_autospec(BaseKnowledgeBase)

    @staticmethod
    @pytest.fixture
    def mock_prompt_builder():
        return create_autospec(BasePromptBuilder)

    @staticmethod
    @pytest.fixture
    def chat_engine(mock_llm,
                    mock_query_builder,
                    mock_knowledge_base,
                    mock_prompt_builder):
        return ChatEngine(
            system_message="system_message",
            llm=mock_llm,
            query_builder=mock_query_builder,
            knowledge_base=mock_knowledge_base,
            prompt_builder=mock_prompt_builder,
            max_prompt_tokens=100,
            max_generated_tokens=200
        )

    @staticmethod
    def test_chat_default_params(chat_engine,
                                 mock_query_builder,
                                 mock_knowledge_base,
                                 mock_prompt_builder,
                                 mock_llm):
        # Mock input and expected output
        messages = [MessageBase(role=Role.USER,
                                content="How does photosynthesis work?")]
        mock_queries = [Query(text="How does photosynthesis work?")]
        mock_query_results = [QueryResult(query="How does photosynthesis work?",
                                          documents=[])]
        mock_prompt_messages = [MessageBase(role=Role.SYSTEM,
                                            content="Some content...")]
        mock_chat_response = "Photosynthesis is a process used by plants..."

        # Set the return values of the mocked methods
        mock_query_builder.generate.return_value = mock_queries
        mock_knowledge_base.query.return_value = mock_query_results
        mock_prompt_builder.build.return_value = mock_prompt_messages
        mock_llm.chat_completion.return_value = mock_chat_response

        # Call the method under test
        response = chat_engine.chat(messages)

        # Assertions
        mock_query_builder.generate.assert_called_once_with(messages,
                                                            max_prompt_tokens=100)
        mock_knowledge_base.query.assert_called_once_with(mock_queries)
        mock_prompt_builder.build.assert_called_once_with(
            "system_message", messages, mock_query_results,
            max_tokens=100)
        mock_llm.chat_completion.assert_called_once_with(
            mock_prompt_messages,
            max_tokens=200,
            stream=False,
            model_params=None
        )

        assert response == mock_chat_response

    @staticmethod
    def test_chat_special_params(chat_engine,
                                 mock_query_builder,
                                 mock_knowledge_base,
                                 mock_prompt_builder,
                                 mock_llm):
        # Mock input and expected output
        messages = [MessageBase(role=Role.USER,
                                content="How does photosynthesis work?")]
        mock_queries = [Query(text="How does photosynthesis work?")]
        mock_query_results = [QueryResult(query="How does photosynthesis work?",
                                          documents=[])]
        mock_prompt_messages = [MessageBase(role=Role.SYSTEM,
                                            content="Some content...")]
        mock_chat_response = "Photosynthesis is a process used by plants..."

        # Set the return values of the mocked methods
        mock_query_builder.generate.return_value = mock_queries
        mock_knowledge_base.query.return_value = mock_query_results
        mock_prompt_builder.build.return_value = mock_prompt_messages
        mock_llm.chat_completion.return_value = mock_chat_response

        model_params = ModelParams(temperature=0.5)
        max_tokens = 300

        # Call the method under test
        response = chat_engine.chat(messages,
                                    stream=True,
                                    max_tokens=max_tokens,
                                    model_params=model_params)

        # Assertions
        mock_query_builder.generate.assert_called_once_with(messages,
                                                            max_prompt_tokens=100)
        mock_knowledge_base.query.assert_called_once_with(mock_queries)
        mock_prompt_builder.build.assert_called_once_with(
            "system_message", messages, mock_query_results, max_tokens=100)
        mock_llm.chat_completion.assert_called_once_with(
            mock_prompt_messages,
            max_tokens=max_tokens,
            stream=True,
            model_params=model_params,
        )

        assert response == mock_chat_response

    @staticmethod
    def test_get_context_raise(chat_engine):
        with pytest.raises(NotImplementedError):
            chat_engine.get_context([])

    @staticmethod
    @pytest.mark.asyncio
    async def test_aget_context_raise(chat_engine):
        with pytest.raises(NotImplementedError):
            await chat_engine.aget_context([])

    @staticmethod
    @pytest.mark.asyncio
    async def test_achat_raise(chat_engine):
        with pytest.raises(NotImplementedError):
            await chat_engine.achat([])
