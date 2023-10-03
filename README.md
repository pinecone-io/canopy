# Resin

**Resin** is a tool that allows you to build AI applications using your own data. It is built on top of Pinecone, the world's most powerfull vector database. **Resin** is desinged to be well packaged and easy to use. It can be used as a library or as a service. It is also designed to be modular, so you can use only the parts that you need. **Resin** is built with the following principles in mind:

* **Easy to use** - **Resin** is designed to be easy to use. It is well packaged and can be installed with a single command. It is also designed to be modular, so you can use only the parts that you need.

* **Production ready** - **Resin** is built on top of Pinecone, the world's most powerfull vector database. It is designed to be production ready, 100% tested, well documented, maintained and supported.

* **Open source** - **Resin** is open source and free to use. It is also designed to be open and extensible, so you can easily add your own components and extend the functionality.

* **Operative AI** - **Resin** is designed to be used in production and as such, it allows developers to set up operating point to have better control over token consumption in prompt or in generation. **Resin** can maximize context quality and relevance while controlling the cost of the AI system.

## Concept

**Resin** can be used as a library and as-a-service. The conceptual model is the following:

![conceptual model](https://github.com/pinecone-io/context-engine/blob/dev/.readme-content/sketch.png)

Where:

* **ChatEngine** _`/context/chat/completions`_ - is a complete RAG unit, this will perform the RAG pipeline and return the answer from the LLM. This API follows the OpenAI API specification and can be used as a drop-in replacement for the OpenAI API.

* **ContextEngine** _`/context/query`_ - is a proxy between your application and Pinecone. It will handle the R in the RAG pipeline and will return the snippet of context along with the respected source. ContextEngine internally performs a process of ContextBuilding - i.e. it will find the most relevant documents to your query and will return them as a single context object.

* **KnowledgeBase** _`/context/upsert`_ - is the interface to upload your data into Pinecone. It will create a new index and will configure it for you. It will also handle the processing of the data, namely - processing, chunking and encoding (embedding).

## How to install

1. Set up the environment variables

```bash
export PINECONE_API_KEY="<PINECONE_API_KEY>"
export PINECONE_ENVIRONMENT="<PINECONE_ENVIRONMENT>"
export OPENAI_API_KEY="<OPENAI_API_KEY>"
export INDEX_NAME="test-index-1"
```

2. install the package
```bash
pip install pinecone-resin
```

3. you are good to go! see the quickstart guide on how to run basic demo

## Quickstart

In this quickstart, we will show you how to use the **Resin** to build a simple question answering system using RAG (retrival augmented generation).

### 0. Before we start

Before we start, run the `resin` command alone to verify the connection to services in healthy:
    
```bash
resin
```

output should be similar to this:

```bash
Resin: Ready

Usage: resin [OPTIONS] COMMAND [ARGS]...
# rest of the help message
```


### 1. Create a new **Resin** Index

**Resin** will create and configure a new Pinecone index on your behalf. Just run:

```bash
resin new
```

And follow the CLI instructions. The index that will be created will have a prefix `resin--<INDEX_NAME>`.

> Note, this will have to be done only once per index.

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/resin-new.gif)

### 2. Uploading data

You can load data into your **Resin** Index by simply using the CLI:

```bash
resin upsert <PATH_TO_DATA>
```

The data should be in a parquet format where each row is a document. The documents should have the following schema:

```
+----------+--------------+--------------+---------------+
| id(str)  | text(str)    | source(str)  | metadata(dict)|
|----------+--------------+--------------+---------------|
| "1"      | "some text"  | "some source"| {"key": "val"}|
+----------+--------------+--------------+---------------+
```

Follow the instructions in the CLI to upload your data.

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/resin-upsert.gif)

### 3. Start the **Resin** service

**Resin** service serve as a proxy between your application and Pinecone. It will also handle the RAG part of the application. To start the service, run:

```bash
resin start
```

Now, you should be prompted with standard Uvicorn logs:

```
Starting Resin service on 0.0.0.0:8000
INFO:     Started server process [24431]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/resin-start.gif)


### 4. Chat with your data

Now that you have data in your index, you can chat with it using the CLI:

```bash
resin chat
```

This will open a chat interface in your terminal. You can ask questions and the **Resin** will try to answer them using the data you uploaded.

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/resin-chat.gif)

To compare the chat response with and without RAG use the `--no-rag` flag

```bash
resin chat --no-rag
```

This will open a similar chat interface window, but will send your question directly to the LLM without the RAG pipeline.

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/resin-chat-no-rag.gif)


### 5. Stop the **Resin** service

To stop the service, simply press `CTRL+C` in the terminal where you started it.

If you have started the service in the background, you can stop it by running:

```bash
resin stop
```

## Advanced usage

### 1. Migrating existing OpenAI application to **Resin**

If you already have an application that uses the OpenAI API, you can migrate it to **Resin** by simply changing the API endpoint to `http://host:port/context` as follows:

```python
import openai

openai.api_base = "http://host:port/context"

# now you can use the OpenAI API as usual
```

or without global state change:

```python
import openai

openai_response = openai.Completion.create(..., api_base="http://host:port/context")
```
