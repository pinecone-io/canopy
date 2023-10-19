# Resin Library

Resin can act both as a library and as a service. This document describes how to use Resin as a library. To read more about Resin project in general and also how to use it as a service, please refer to the [main README](../README.md).


> 💡 You can find notebooks with examples of how to use Resin library [here](../examples).

The idea behind Resin library is to provide a framework to build AI applications on top of Pinecone as a long memory storage for you own data. Resin library designed with the following principles in mind:

- **Easy to use**: Resin is designed to be easy to use. It is well packaged and can be installed with a single command.
- **Modularity**: Resin is built as a collection of modules that can be used together or separately. For example, you can use the `chat_engine` module to build a chatbot on top of your data, or you can use the `knowledge_base` module to directly store and search your data.
- **Extensibility**: Resin is designed to be extensible. You can easily add your own components and extend the functionality.
- **Production ready**: Resin designed to be production ready, tested, well documented, maintained and supported.
- **Open source**: Resin is open source and free to use. It built in partnership with the community and for the community.


## High level architecture

TBD

## Installation

To install Resin, you only need to run a single command:

```bash
pip install pinecone-resin
```

It is highly recommended to use a virtual environment for you project. You can read more about virtual environments [here](https://docs.python.org/3/tutorial/venv.html).

## Quickstart


### Step 0: Set up the environment variables

Before you start using Resin, you need to set up the following environment variables:

```bash
export PINECONE_API_KEY="<PINECONE_API_KEY>"
export PINECONE_ENVIRONMENT="<PINECONE_ENVIRONMENT>"
export OPENAI_API_KEY="<OPENAI_API_KEY>"
```

<details>
<summary>CLICK HERE FOR MORE DETAILS</summary>

| Name                  | Description                                                                                                                 | How to get it?                                                                                                                                                               |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `PINECONE_API_KEY`    | The API key for Pinecone. Used to authenticate to Pinecone services to create indexes and to insert, delete and search data | Register or log into your Pinecone account in the [console](https://app.pinecone.io/). You can access your API key from the "API Keys" section in the sidebar of your dashboard |
| `PINECONE_ENVIRONMENT`| Determines the Pinecone service cloud environment of your index e.g `west1-gcp`, `us-east-1-aws`, etc                       | You can find the Pinecone environment next to the API key in [console](https://app.pinecone.io/)                                                                             |
| `OPENAI_API_KEY`      | API key for OpenAI. Used to authenticate to OpenAI's services for embedding and chat API                                    | You can find your OpenAI API key [here](https://platform.openai.com/account/api-keys). You might need to login or register to OpenAI services                                |
</details>


### Step 1: Initialize global Tokenizer

Tokenizer is an object that is responsible for splitting text into tokens, which are the basic units of text that are used for processing.

Resin uses a global tokenizer for chunking, token counting and more. You can initialize it with the following command:

```python
from resin.tokenizer import Tokenizer
Tokenizer.initialize()
```

Then, each time you want to use the tokenizer, you can simply initialize it with the following command:

```python
from resin.tokenizer import Tokenizer

# no need to pass any parameters, the global tokenizer will be used
tokenizer = Tokenizer()

tokenizer.tokenize("Hello world!")
# output: ['Hello', 'world', '!']
```

By default, the global tokenizer is initialized with `OpenAITokenizer` that is based on OpenAI's tiktoken library and aligned with GPT 3 and 4 models tokenization.

<details>
<summary>Go deeper</summary>
The global tokenizer is holding an underlying tokenizer that implements `BaseTokenizer`.

You can add a customized tokenizer of your own by implement a subclass of `BaseTokenizer` and pass the class in the `tokenizer_class` parameter.

To use additional parameters to init the underlying tokenizer, you can simply pass them to the initialize method of the global tokenizer. For example:

```python
from resin.tokenizer import Tokenizer
from resin.tokenizer.openai import OpenAITokenizer
Tokenizer.initialize(tokenizer_class=OpenAITokenizer, model_name="gpt2")
```

Will initialize the global tokenizer with `OpenAITokenizer` and will pass the `model_name` parameter to the underlying tokenizer.
</details>


### Step 2: Create a knowledge base
Knowledge base is an object that is responsible for storing and query your data. It holds a connection to a single Pinecone index and provides a simple API to insert, delete and search textual documents.

To create a knowledge base, you can use the following command:

```python
from resin.knowledge_base import KnowledgeBase

kb = KnowledgeBase(index_name="my-index")
```

To create a new Pinecone index and connect it to the knowledge base, you can use the `create_resin_index` method:

```python
kb.create_resin_index()
```

Then, you will be able to mange the index in Pinecone [console](https://app.pinecone.io/).

If you already created a Pinecone index, you can connect it to the knowledge base with the `connect` method:

```python
kb.connect()
```

You can always verify the connection to the Pinecone index with the `verify_index_connection` method:

```python
kb.verify_index_connection()
```


<details>
<summary>Go deeper</summary>
TBD
</details>


### Step 3: Upsert and query data

To insert data into the knowledge base, you can create a list of documents and use the `upsert` method:

```python
from resin.models.data_models import Document
documents = [Document(id="1", text="U2 are an Irish rock band from Dublin, formed in 1976.", source="https://url.com"),
             Document(id="2", text="Arctic Monkeys are an English rock band formed in Sheffield in 2002.", source="https://another-url.com", metadata={"my-key": "my-value"})]
kb.upsert(documents)
```

Now you can query the knowledge base with the `query` method to find the most similar documents to a given text:

```python
from resin.models.query_models import Query
results = kb.query([Query("Arctic Monkeys music genre")])

print(results[0].documents[0].text)
# output: Arctic Monkeys are an English rock band formed in Sheffield in 2002.
```

<details>
<summary>Go deeper</summary>
TBD
</details>

### Step 4: Create a context engine

Context engine is an object that responsible to retrieve the most relevant context for a given query and token budget. It uses the knowledge base to retrieve the most relevant documents and then constructs a context that does not exceed the token budget.
The output of the context engine designed to interact with LLMs and try to provide the LLM with the most relevant context for a given query, while ensuring that the context does not exceed the prompt boundary.


To create a context engine using a knowledge base, you can use the following command:

```python
from resin.context_engine import ContextEngine
context_engine = ContextEngine(kb)
```

Then, you can use the `query` method to retrieve the most relevant context for a given query and token budget:

```python
result = context_engine.query([Query("Arctic Monkeys music genre")], token_budget=100)

print(result.content)
# output: Arctic Monkeys are an English rock band formed in Sheffield in 2002.

print(result.token_count)
# output: 17
```

By default, to handle the token budget constraint, the context engine will use the `StuffingContextBuilder` that will stuff as many documents as possible into the context without exceeding the token budget, by the order they have been retrieved from the knowledge base.


<details>
<summary>Go deeper</summary>
TBD
</details>


### Step 5: Create a chat engine

Chat engine is an object that implements end to end chat API with [RAG](https://www.pinecone.io/learn/retrieval-augmented-generation/).
Given chat history, the chat engine orchestrates its underlying context engine and LLM to run the following steps:

1. Generate search queries from the chat history
2. Retrieve the most relevant context for each query using the context engine
3. Prompt the LLM with the chat history and the retrieved context to generate the next response

To create a chat engine using a context, you can use the following command:

```python
from resin.chat_engine import ChatEngine
chat_engine = ChatEngine(context_engine)
```

Then, you can start chatting!

```python
chat_engine.chat("what is the genre of Arctic Monkeys band?")
# output: Arctic Monkeys is a rock band.
```


Resin designed to be production ready and handle any conversation length and context length. Therefore, the chat engine uses internal components to handle long conversations and long contexts.
By default, long chat history is truncated to the latest messages that fits the token budget. It orchestrates the context engine to retrieve context that fits the token budget and then use the LLM to generate the next response.


<details>
<summary>Go deeper</summary>
TBD
</details>


