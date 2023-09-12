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

![](https://github.com/pinecone-io/context-engine/blob/add_cli/.readme-content/new.gif)
