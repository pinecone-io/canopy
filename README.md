# Resin

**Resin** is a Sofware Development Kit (SDK) for AI applications. Resin allows you to test, build and package Retrieval Augmented Applications with Pinecone Vector Database. **Resin** is desinged to be well packaged and easy to use. It can be used as-a-library or as-a-service and designed to be modular, so you can use only the parts that you need. **Resin** ships with a developer friendly CLI, to help you kickoff and test your application quickly.

## RAG with Resin

**Pinecone + LLM = ❤️** 

By enhancing language models with access to unlearned knowledge and inifinite memory we can build AI applications that can answer questions and assist humans without the risk of hallucinating or generating fake content. Let's learn how Resin executes RAG pipeline.

![](.readme-content/rag_flow.png)

<details>
<summary><b>Chat Flow</b> (click to expand)
</summary>

1. User will promt a question to Resin /chat/completions endpoint. 
2. resin will use a language model to break down the questions into queries, sometimes, a single user ask may result in multiple knowledge queries. 
3. Resin will encode and embed each query seperateley.
4. Resin will query pinecone with the embedded queries and will fetch back K results for each. Resin will determine how many results it needs to fetch based on the token budget set by the user 
5. Now resin has all the external knowledge needed to answer the original question, Resin will perform a _context building_ step to create an on-bugdet optimal context.
6. Resin will generate a prompt combining general task information and the system message and sent the prompt+context to the language model. 
7. Resin will decode the response from the language model and will return the response in the API response (or in streaming).

</details>

<details>
<summary><b>Context Flow</b> (click to expand)
</summary>

<ol type="I">
<li> User will call /context/upsert with Documents - each document with id, text, and optinally source and metadata </li>

<li> Resin KnowledgeBase will process the documents and chunk ecah document in a structural and semantic way </li>

<li> Resin KnowledgeBase will encode each chunk using one or more embedding models</li>

<li> Resin KnowledgeBase will upsert the encoded chunks into Pinecone Index</li>

</ol>
</details>

## Why Resin? [TODO: TBD]

* **Ease of use** - Installed with a single command and can deploy an AI application in minutes. **Resin** is designed to be easy to use and easy to integrate with your existing applications and compatible with OpenAI /chat/completions API. 

* **Operational & Production ready** - Resin is built on top of **Pinecone** and can Scale to Billions of documents. Unlike other AI frameworks, **Resin** optimizes for production use cases it allows developers to set up operating point to have better control over token consumption in prompt or in generation. **Resin** can maximize context quality and relevance while controlling the cost of the AI system.

* **Open source** - **Resin** is open source and free to use. It is also designed to be open and extensible, so you can easily add your own components and extend the functionality.


## What's inside the box?

1. **Resin Core Library** - Resin has 3 high level classes that act as API level components:
    * **ChatEngine** _`/chat/completions`_  - is a complete RAG unit that exposes a chat interface of LLM augmented with retrieval engine.
    * **ContextEngine** _`/context/query`_ - is a proxy between your application and Pinecone. It will handle the R in the RAG pipeline and will return the snippet of context along with the respected source. 
    * **KnowledgeBase** _`/context/{upsert, delete}` -  is the data managment interface, handles the processing, chunking and encoding (embedding) of the data, along with Upsert and Delete operations

> more information about the Core Library usage can be found in the [Library Documentation](docs/library.md)

2. **Resin Service** - a webservice that wraps the **Resin Core** and exposes it as a REST API. The service is built on top of FastAPI, Uvicorn and Gunicorn and can be easily deployed in production. 

> For the complete documentation please go to: [#TODO: LINK](link.link.com) 

3. **Resin CLI** - Resin comes with a fully functional CLI that is purposley built to allow users to quickly test their configuration and application before shipping, the CLI also comes with managment operations that allow you to create indexes and load data quickly

## Setup

0. set up a virtual environment (optional)
```bash
python3 -m venv resin-env
source resin-env/bin/activate
```
more about virtual environments [here](https://docs.python.org/3/tutorial/venv.html)

1. install the package
```bash
pip install pinecone-resin
```

2. Set up the environment variables

```bash
export PINECONE_API_KEY="<PINECONE_API_KEY>"
export PINECONE_ENVIRONMENT="<PINECONE_ENVIRONMENT>"
export OPENAI_API_KEY="<OPENAI_API_KEY>"
export INDEX_NAME=<INDEX_NAME>
```

<details>
<summary><b><u>CLICK HERE</u></b> for more information about the environment variables 

<br /> 
</summary>

| Name                  | Description                                                                                                                 | How to get it?                                                                                                                                                               |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `PINECONE_API_KEY`    | The API key for Pinecone. Used to authenticate to Pinecone services to create indexes and to insert, delete and search data | Register or log into your Pinecone account in the [console](https://app.pinecone.io/). You can access your API key from the "API Keys" section in the sidebar of your dashboard |
| `PINECONE_ENVIRONMENT`| Determines the Pinecone service cloud environment of your index e.g `west1-gcp`, `us-east-1-aws`, etc                       | You can find the Pinecone environment next to the API key in [console](https://app.pinecone.io/)                                                                             |
| `OPENAI_API_KEY`      | API key for OpenAI. Used to authenticate to OpenAI's services for embedding and chat API                                    | You can find your OpenAI API key [here](https://platform.openai.com/account/api-keys). You might need to login or register to OpenAI services                                |
| `INDEX_NAME`          | Name of the Pinecone index Resin will underlying work with                                                                  | You can choose any name as long as it follows Pinecone's [restrictions](https://support.pinecone.io/hc/en-us/articles/11729246212637-Are-there-restrictions-on-index-names-#:~:text=There%20are%20two%20main%20restrictions,and%20emojis%20are%20not%20supported.)                                                                                       |
| `RESIN_CONFIG_FILE` | The path of a configuration yaml file to be used by the Resin service. | Optional - if not provided, default configuration would be used |
</details>


3. Check that installation is successful and environment is set, run:
```bash
resin
```

output should be similar to this:

```bash
Resin: Ready

Usage: resin [OPTIONS] COMMAND [ARGS]...
# rest of the help message
```

## Quickstart

In this quickstart, we will show you how to use the **Resin** to build a simple question answering system using RAG (retrival augmented generation).

### 1. Create a new **Resin** Index

**Resin** will create and configure a new Pinecone index on your behalf. Just run:

```bash
resin new
```

And follow the CLI instructions. The index that will be created will have a prefix `resin--<INDEX_NAME>`. This will have to be done only once per index.

> To learn more about Pinecone Indexes and how to manage them, please refer to the following guide: [Understanding indexes](https://docs.pinecone.io/docs/indexes)

![](.readme-content/resin-new.gif)

### 2. Uploading data

You can load data into your **Resin** Index by simply using the CLI:

```bash
resin upsert /path/to/data_directory

# or
resin upsert /path/to/data_directory/file.parquet

# or
resin upsert /path/to/data_directory/file.jsonl
```

Resin support single or mulitple files in jsonl or praquet format. The documents should have the following schema:

```
+----------+--------------+--------------+---------------+
| id(str)  | text(str)    | source       | metadata      |
|          |              | Optional[str]| Optional[dict]|
|----------+--------------+--------------+---------------|
| "id1"    | "some text"  | "some source"| {"key": "val"}|
+----------+--------------+--------------+---------------+
```

Follow the instructions in the CLI to upload your data.

![](.readme-content/resin-upsert.gif)

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

![](.readme-content/resin-start.gif)


> **_📝 NOTE:_**
>
> The resin start command will keep the terminal occupied. To proceed with the next steps, please open a new terminal window.
> and make sure all the environment variables described in the [installation](#how-to-install) section are set.
> If you want to run the service in the background, you can use the following command - **```nohup resin start &```**


### 4. Chat with your data

Now that you have data in your index, you can chat with it using the CLI:

```bash
resin chat
```

This will open a chat interface in your terminal. You can ask questions and the **Resin** will try to answer them using the data you uploaded.

![](.readme-content/resin-chat.gif)

To compare the chat response with and without RAG use the `--baseline` flag

```bash
resin chat --baseline
```

This will open a similar chat interface window, but will send your question directly to the LLM without the RAG pipeline.

![](.readme-content/resin-chat-no-rag.gif)


### 5. Stop the **Resin** service

To stop the service, simply press `CTRL+C` in the terminal where you started it.

If you have started the service in the background, you can stop it by running:

```bash
resin stop
```

## Advanced usage

### Migrating existing OpenAI application to **Resin**

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

### Running Resin service in production

Resin is using FastAPI as the web framework and Uvicorn as the ASGI server. It is recommended to use Gunicorn as the production server, mainly because it supports multiple worker processes and can handle multiple requests in parallel, more details can be found [here](https://www.uvicorn.org/deployment/#using-a-process-manager).

To run the resin service for production, please run:

```bash
gunicorn resin_cli.app:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers <number of desired worker processes>
```