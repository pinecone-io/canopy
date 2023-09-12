# context-engine

## How to install and run the demo

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
export INDEX_NAME_SUFFIX="test-index-1"
```

> NOTE -- the INDEX_NAME_SUFFIX and the index name you use in the CLI operaions should match (i.e. if you use `context-engine new test-index-1` in the CLI, then the INDEX_NAME_SUFFIX should be `test-index-1`). AGAIN THIS IS TEMPORARY AND WILL BECOME OBSOLETE SOON!

3. install the dependencies
```bash
pip install -e .
```


4. you are good to go! see the demo on how to run it