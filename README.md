# Canopy

**Canopy** is a Sofware Development Kit and a Framework for AI applications. Canopy allows you to test, build and package Retrieval Augmented Applications with Pinecone Vector Database. **Canopy** is desinged to be well packaged and easy to use. It can be used as-a-library or as-a-service and designed to be modular, so you can use only the parts that you need. 

* **Ease of use** - Installed with a single command and can deploy an AI application in minutes. **Canopy** is designed to be easy to use and easy to integrate with your existing applications and compatible with OpenAI /chat/completions API. 

* **Operational & Production ready** - Canopy is built on top of **Pinecone** and can Scale to billions of vectors. Unlike other AI frameworks, **Canopy** optimizes for production use cases it allows developers to set up operating point to have better control over token consumption in prompt or in generation. **Canopy** can maximize context quality and relevance while controlling the cost of the AI system.

* **Open source** - **Canopy** is open source and free to use. It is also designed to be open and extensible, so you can easily add your own components and extend the functionality.


## What's inside the box?

1. **Canopy Core** - the core library. Canopy has 3 high level classes that act as API level components:
    * **ChatEngine** - is a complete RAG unit [TBD]
    * **ContextEngine** - is a proxy between your application and Pinecone. It will handle the R in the RAG pipeline and will return the snippet of context along with the respected source. 
    * **KnowledgeBase** - is the data managment interface, handles the processing, chunking and encoding (embedding) of the data, along with Upsert/Query and Delete operations

2. **Canopy Service** - a service that wraps the **Canopy Core** and exposes it as a REST API. The service is built on top of FastAPI and Uvicorn and can be easily deployed in production. 

3. **Canopy CLI** - [TBD]


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
pip install pinecone-canopy
```

3. you are good to go! see the quickstart guide on how to run basic demo

## Quickstart

In this quickstart, we will show you how to use the **Canopy** to build a simple question answering system using RAG (retrival augmented generation).

### 0. Before we start

Before we start, run the `canopy` command alone to verify the connection to services in healthy:
    
```bash
canopy
```

output should be similar to this:

```bash
Canopy: Ready

Usage: canopy [OPTIONS] COMMAND [ARGS]...
# rest of the help message
```


### 1. Create a new **Canopy** Index

**Canopy** will create and configure a new Pinecone index on your behalf. Just run:

```bash
canopy new
```

And follow the CLI instructions. The index that will be created will have a prefix `canopy--<INDEX_NAME>`.

> Note, this will have to be done only once per index.

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/canopy-new.gif)

### 2. Uploading data

You can load data into your **Canopy** Index by simply using the CLI:

```bash
canopy upsert <PATH_TO_DATA>
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

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/canopy-upsert.gif)

### 3. Start the **Canopy** service

**Canopy** service serve as a proxy between your application and Pinecone. It will also handle the RAG part of the application. To start the service, run:

```bash
canopy start
```

Now, you should be prompted with standard Uvicorn logs:

```
Starting Canopy service on 0.0.0.0:8000
INFO:     Started server process [24431]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/canopy-start.gif)


### 4. Chat with your data

Now that you have data in your index, you can chat with it using the CLI:

```bash
canopy chat
```

This will open a chat interface in your terminal. You can ask questions and the **Canopy** will try to answer them using the data you uploaded.

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/canopy-chat.gif)

To compare the chat response with and without RAG use the `--no-rag` flag

```bash
canopy chat --no-rag
```

This will open a similar chat interface window, but will send your question directly to the LLM without the RAG pipeline.

![](https://github.com/pinecone-io/context-engine/blob/change-readme-cli-names/.readme-content/canopy-chat-no-rag.gif)


### 5. Stop the **Canopy** service

To stop the service, simply press `CTRL+C` in the terminal where you started it.

If you have started the service in the background, you can stop it by running:

```bash
canopy stop
```

## Advanced usage

### 1. Migrating existing OpenAI application to **Canopy**

If you already have an application that uses the OpenAI API, you can migrate it to **Canopy** by simply changing the API endpoint to `http://host:port/context` as follows:

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
