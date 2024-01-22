import json

import pytest
from unittest.mock import create_autospec

from canopy.context_engine import ContextEngine
from canopy.context_engine.context_builder.base import ContextBuilder
from canopy.context_engine.context_builder.stuffing import (ContextSnippet,
                                                            ContextQueryResult,
                                                            StuffingContextContent, )
from canopy.knowledge_base.base import BaseKnowledgeBase
from canopy.knowledge_base.models import QueryResult, DocumentWithScore
from canopy.models.data_models import Query, Context, ContextContent


@pytest.fixture
def mock_knowledge_base():
    return create_autospec(BaseKnowledgeBase)


@pytest.fixture
def mock_context_builder():
    return create_autospec(ContextBuilder)


@pytest.fixture
def context_engine(mock_knowledge_base, mock_context_builder):
    return ContextEngine(knowledge_base=mock_knowledge_base,
                         context_builder=mock_context_builder)


@pytest.fixture
def sample_context_text():
    return (
        "Photosynthesis is the process used by plants, algae and certain bacteria "
        "to harness energy from sunlight and turn it into chemical energy."
    )


@pytest.fixture
def mock_global_metadata_filter():
    return {"sourcerer": "Wikipedia"}


@pytest.fixture
def mock_query_result(sample_context_text):
    return [
        QueryResult(
            query="How does photosynthesis work?",
            documents=[
                DocumentWithScore(
                    id="1",
                    text=sample_context_text,
                    metadata={"sourcerer": "Wikipedia"},
                    score=0.95
                )
            ]
        )
    ]


def test_query(context_engine,
               mock_knowledge_base,
               mock_context_builder,
               sample_context_text,
               mock_query_result,
               namespace):
    queries = [Query(text="How does photosynthesis work?")]
    max_context_tokens = 100

    mock_context_content = create_autospec(ContextContent)
    mock_context_content.to_text.return_value = sample_context_text
    mock_context = Context(content=mock_context_content, num_tokens=21)

    mock_knowledge_base.query.return_value = mock_query_result
    mock_context_builder.build.return_value = mock_context

    result = context_engine.query(queries, max_context_tokens, namespace=namespace)

    assert result == mock_context
    mock_knowledge_base.query.assert_called_once_with(
        queries, global_metadata_filter=None, namespace=namespace)
    mock_context_builder.build.assert_called_once_with(
        mock_query_result, max_context_tokens)


def test_query_with_metadata_filter(context_engine,
                                    mock_knowledge_base,
                                    mock_context_builder,
                                    sample_context_text,
                                    mock_query_result,
                                    mock_global_metadata_filter,
                                    namespace):
    queries = [Query(text="How does photosynthesis work?")]
    max_context_tokens = 100

    mock_context_content = create_autospec(ContextContent)
    mock_context_content.to_text.return_value = sample_context_text
    mock_context = Context(content=mock_context_content, num_tokens=21)

    mock_knowledge_base.query.return_value = mock_query_result
    mock_context_builder.build.return_value = mock_context

    context_engine_with_filter = ContextEngine(
        knowledge_base=mock_knowledge_base,
        context_builder=mock_context_builder,
        global_metadata_filter=mock_global_metadata_filter
    )

    result = context_engine_with_filter.query(queries, max_context_tokens,
                                              namespace=namespace)

    assert result == mock_context
    mock_knowledge_base.query.assert_called_once_with(
        queries, global_metadata_filter=mock_global_metadata_filter,
        namespace=namespace)
    mock_context_builder.build.assert_called_once_with(
        mock_query_result, max_context_tokens)


def test_multiple_queries(context_engine,
                          mock_knowledge_base,
                          mock_context_builder,
                          sample_context_text,
                          mock_query_result):
    queries = [
        Query(text="How does photosynthesis work?"),
        Query(text="What is cellular respiration?")
    ]
    max_context_tokens = 200

    text = (
        "Cellular respiration is a set of metabolic reactions and processes "
        "that take place in the cells of organisms to convert biochemical energy "
        "from nutrients into adenosine triphosphate (ATP)."
    )

    extended_mock_query_result = mock_query_result + [
        QueryResult(
            query="What is cellular respiration?",
            documents=[
                DocumentWithScore(
                    id="2",
                    text=text,
                    metadata={"sourcerer": "Wikipedia"},
                    score=0.93
                )
            ]
        )
    ]

    mock_knowledge_base.query.return_value = extended_mock_query_result

    combined_text = sample_context_text + "\n" + text
    mock_context_content = create_autospec(ContextContent)
    mock_context_content.to_text.return_value = combined_text
    mock_context = Context(content=mock_context_content, num_tokens=40)

    mock_context_builder.build.return_value = mock_context

    result = context_engine.query(queries, max_context_tokens)

    assert result == mock_context


def test_empty_query_results(context_engine,
                             mock_knowledge_base,
                             mock_context_builder):
    queries = [Query(text="Unknown topic")]
    max_context_tokens = 100

    mock_knowledge_base.query.return_value = []

    mock_context_content = create_autospec(ContextContent)
    mock_context_content.to_text.return_value = ""
    mock_context = Context(content=mock_context_content, num_tokens=0)

    mock_context_builder.build.return_value = mock_context

    result = context_engine.query(queries, max_context_tokens)

    assert result == mock_context


def test_context_query_result_to_text():
    query_result = ContextQueryResult(query="How does photosynthesis work?",
                                      snippets=[ContextSnippet(text="42",
                                                               source="ref")])
    context = Context(content=StuffingContextContent(__root__=[query_result]),
                      num_tokens=1)

    assert context.to_text() == json.dumps([query_result.dict()])
    assert context.to_text(indent=2) == json.dumps([query_result.dict()], indent=2)


@pytest.mark.asyncio
async def test_aquery_not_implemented(context_engine):
    queries = [Query(text="What is quantum physics?")]
    max_context_tokens = 10

    with pytest.raises(NotImplementedError):
        await context_engine.aquery(queries, max_context_tokens)
