# context-engine

Context Engine is a tool that allows you to build RAG applications using your own data. It is built on top of Pinecone, the world's most powerfull vector database.

## How to install

1. clone the repo and cd into it
```bash
   git clone git@github.com:pinecone-io/context-engine.git
   cd context-engine
```
2. checkout the server branch
```bash
   git checkout add_cli
```

3. add .env file with --> THIS STEP WILL BECOME OBSOLETE SOON AND WE WILL HAVE PROPER CONFIGURATION MANAGEMENT
```bash
vi .env
```
add the following envs

```bash
PINECONE_API_KEY="<PINECONE_API_KEY>"
PINECONE_ENVIRONMENT="<PINECONE_ENVIRONMENT>"
OPENAI_API_KEY="<OPENAI_API_KEY>"
INDEX_NAME_SUFFIX="test-index-1"
```

or set them in your shell

```bash
export PINECONE_API_KEY="<PINECONE_API_KEY>"
export PINECONE_ENVIRONMENT="<PINECONE_ENVIRONMENT>"
export OPENAI_API_KEY="<OPENAI_API_KEY>"
export INDEX_NAME="test-index-1"
```

3. install the package
```bash
pip install -e .
```

4. you are good to go! see the demo on how to run it

## Quickstart

In this quickstart, we will show you how to use the Context Engine to build a simple question answering system using RAG (retrival augmented generation).

### 1. Create a new Context Engine Index

Context Engine will create and configure a new Pinecone index on your behalf. Just run:

```bash
context-engine new
```

And follow the CLI instructions. The index that will be created will have a prefix `context-engine-INDEX_NAME`.

> Note, this will have to be done only once per index.

![](https://github.com/pinecone-io/context-engine/blob/add_cli/.readme-content/new.gif)

### 2. Uploading data

You can load data into your Context Engine Index by simply using the CLI:

```bash
context-engine upsert <PATH_TO_DATA>
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

### 3. Start the Context Engine service

Context Engine service serve as a proxy between your application and Pinecone. It will also handle the RAG part of the application. To start the service, run:

```bash
context-engine start
```

Now, you should be prompted with standard Uvicorn logs:

```
Starting Context Engine service on 0.0.0.0:8000
INFO:     Started server process [24431]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 4. Chat with your data

Now that you have data in your index, you can chat with it using the CLI:

```bash
context-engine chat
```

This will open a chat interface in your terminal. You can ask questions and the Context Engine will try to answer them using the data you uploaded.

To compare the chat response with and without RAG use the `--no-rag` flag

```bash
context-engine chat --no-rag
```

This will open a similar chat interface window, but will send your question directly to the LLM without the RAG pipeline.


### 5. Stop the Context Engine service

To stop the service, simply press `CTRL+C` in the terminal where you started it.

If you have started the service in the background, you can stop it by running:

```bash
context-engine stop
```


