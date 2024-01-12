from canopy.chat_engine.query_generator.cohere import CohereQueryGenerator
from canopy.models.data_models import MessageBase, Role


def test_generate_queries():
    query_generator = CohereQueryGenerator()
    messages = [
        MessageBase(
            role=Role.USER, content="Hello, assistant."),
        MessageBase(
            role=Role.ASSISTANT, content="Hello, user. How can I assist you?"),
        MessageBase(
            role=Role.USER, content="How do I init a pinecone client?.")
    ]
    queries = query_generator.generate(messages, max_prompt_tokens=100)
    assert queries
    assert queries[0].text
