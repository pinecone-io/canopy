# Canopy

**Canopy** is a tool that allows you to build AI applications using your own data. **Canopy** is built with the following principles in mind:

* **Easy to use** - **Canopy** is designed to be easy to use. It is well packaged and can be installed with a single command. It is also designed to be modular, so you can use only the parts that you need.

* **Production ready** - **Canopy** is built on top of Pinecone, the world's most powerfull vector database. It is designed to be production ready, 100% tested, well documented, maintained and supported.

* **Open source** - **Canopy** is open source and free to use. It is also designed to be open and extensible, so you can easily add your own components and extend the functionality.

* **Operative AI** - **Canopy** is designed to be used in production. It allows developers to set up operating point to have better control over token consumption in prompt or in generation. **Canopy** can maximize context quality and relevance while controlling the cost of the AI system.

## Concept

**Canopy** can be used as a library and as a service. The following diagram illustrates the conceptual model:

![conceptual model](https://github.com/pinecone-io/context-engine/blob/dev/.readme-content/sketch.png)

In the diagram above, entities have the following meanings:

* **ChatEngine** _`/context/chat/completions`_ - is a complete RAG unit. This performs the RAG pipeline and returns the answer from the LLM. This API follows the OpenAI API specification and can be used as a drop-in replacement for the OpenAI API.

* **ContextEngine** _`/context/query`_ - is a proxy between your application and Pinecone. It handles the retrieval in the RAG pipeline and returns the snippet of context along with the trusted source. ContextEngine internally performs a process of ContextBuilding; that is, it finds the most relevant documents to your query and returns them as a single context object.

* **KnowledgeBase** _`/context/upsert`_ - is the interface to upload your data into Pinecone. It creates a new index and configures it for you. It also handles the processing of the data, namely processing, chunking and encoding (embedding).

## How to install Canopy

Follow the steps below to install Canopy.

1. Set up the environment variables.

```bash
export PINECONE_API_KEY="<PINECONE_API_KEY>"
export PINECONE_ENVIRONMENT="<PINECONE_ENVIRONMENT>"
export OPENAI_API_KEY="<OPENAI_API_KEY>"
export INDEX_NAME="test-index-1"
```

2. Install the package.
   
```bash
pip install pinecone-resin
```

Now you are good to go! See the quickstart guide below on how to run basic demo.

## Quickstart

In this quickstart, we show you how to use **Canopy** to build a simple question-answering system using RAG (retrival augmented generation).

### 1. Create a new **Canopy** index.

**Canopy** creates and configures a new Pinecone index on your behalf. Just run the following command:

```bash
resin new
```

Next, follow the CLI instructions. The new index has the prefix `resin--<INDEX_NAME>`.

> Note: Perform this step only once per index.

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/resin-new.gif)

### 2. Upload data.

You can load data into your **Canopy** Index using the CLI. Run the following command:

```bash
resin upsert <PATH_TO_DATA>
```

The data should be in a Parquet format, where each row is a document. The documents should have the following schema:

```
+----------+--------------+--------------+---------------+
| id(str)  | text(str)    | source(str)  | metadata(dict)|
|----------+--------------+--------------+---------------|
| "1"      | "some text"  | "some source"| {"key": "val"}|
+----------+--------------+--------------+---------------+
```

Follow the instructions in the CLI to upload your data.

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/resin-upsert.gif)

### 3. Start the **Canopy** service.

The **Canopy** service serves as a proxy between your application and Pinecone. It also handles the RAG part of the application. To start the service, run the following command:

```bash
resin start
```

Now, you should be prompted with standard Uvicorn logs, like in the following example shell output:

```
Starting Canopy service on 0.0.0.0:8000
INFO:     Started server process [24431]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/resin-start.gif)


### 4. Chat with your data.

Now that you have data in your index, you can chat with it using the CLI. Run the following command:

```bash
resin chat
```

This will open a chat interface in your terminal. If you ask questions, the **Canopy** service tries to answer them using the data you uploaded.

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/resin-chat.gif)

To compare the chat response with and without RAG, use the `--no-rag` flag, as in the following command:

```bash
resin chat --no-rag
```

This opens a similar chat interface window, but sends your question directly to the LLM without the RAG pipeline.

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/resin-chat-no-rag.gif)


### 5. Stop the **Canopy** service

To stop the service, press `CTRL+C` in the terminal where you started it.

If you have started the service in the background, you can stop it by running the following command:

```bash
resin stop
```

## Advanced usage

### 1. Migrating existing OpenAI applications to **Canopy**

If you already have an application that uses the OpenAI API, you can migrate it to **Canopy** by changing the API endpoint to `http://host:port/context` as follows:

```python
import openai

openai.api_base = "http://host:port/context"

# now you can use the OpenAI API as usual
```

To perform the same task without global state change, use code like the following:

```python
import openai

openai_response = openai.Completion.create(..., api_base="http://host:port/context")
```
