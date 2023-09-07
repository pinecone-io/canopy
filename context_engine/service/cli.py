import os
import click
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate

from context_engine.knoweldge_base import KnowledgeBase
from context_engine.knoweldge_base.knowledge_base import INDEX_NAME_PREFIX
from context_engine.knoweldge_base.tokenizer import OpenAITokenizer, Tokenizer
from context_engine.context_engine import ContextEngine
from context_engine.chat_engine import ChatEngine
from context_engine.llm.openai import OpenAILLM

from context_engine.models.data_models import Query, Messages
from context_engine.llm.models import UserMessage
from context_engine.service.app import start as start_service


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)


@click.group(help="CLI for the Context Engine")
def cli():
    pass

    
@cli.command()
@click.argument('index-name-suffix', nargs=1)
@click.option('--tokenizer-model', default="gpt-3.5-turbo", help='Tokenizer model')
def new(index_name_suffix, tokenizer_model):
    click.echo(f"Context Engine is going to create a new index: {INDEX_NAME_PREFIX}{index_name_suffix}")
    click.confirm(click.style('Do you want to continue?', fg="red"), abort=True)
    Tokenizer.initialize(OpenAITokenizer, tokenizer_model)
    kb = KnowledgeBase(index_name_suffix=index_name_suffix)
    kb.create_index()
    click.echo(click.style("Success!", fg="green"))
    os.environ["INDEX_NAME_SUFFIX"] = index_name_suffix


@cli.command()
@click.argument('data-path', type=click.Path(exists=True))
@click.option('--index-name-suffix', default=os.environ.get("INDEX_NAME_SUFFIX"), help='Index name suffix')
@click.option('--tokenizer-model', default="gpt-3.5-turbo", help='Tokenizer model')
def upsert(index_name_suffix, data_path, tokenizer_model):
    if index_name_suffix is None:
        raise ValueError("Must provide index name suffix")
    Tokenizer.initialize(OpenAITokenizer, tokenizer_model)
    if data_path is None:
        raise ValueError("Must provide data path")
    click.echo(f"Context Engine is going to upsert data from {data_path} to index: {INDEX_NAME_PREFIX}{index_name_suffix}")
    kb = KnowledgeBase(index_name_suffix=index_name_suffix)
    kb.connect()
    click.echo("")
    data = pd.read_parquet(data_path)
    click.echo(tabulate(data.head(), headers='keys', tablefmt='psql', showindex=False))
    click.confirm(click.style('Does this data look right?', fg="red"), abort=True)
    kb.upsert_dataframe(data)
    click.echo(click.style("Success!", fg="green"))


@cli.command()
@click.argument('message', nargs=-1)
@click.option('--index-name-suffix', default=os.environ.get("INDEX_NAME_SUFFIX"), help='Index name suffix')
@click.option('--max-prompt-tokens', default=4096, help='Max prompt tokens that will be sent at any point in time to the model')
@click.option('--max-context-tokens', default=2048, help='Max context tokens that will be sent at any point in time to the model')
@click.option('--max-generated-tokens', default=128, help='Max generated tokens that will be sent at any point in time to the model')
@click.option('--llm-model-name', default="gpt-3.5-turbo", help='LLM model name')
@click.option('--tokenizer-model', default="gpt-3.5-turbo", help='Tokenizer model')
def chat(index_name_suffix, message, max_prompt_tokens, max_context_tokens, max_generated_tokens, llm_model_name, tokenizer_model):
    if index_name_suffix is None:
        raise ValueError("Index was not initialized in this session, if you want to chat with existing index pass --index-name-suffix parameter")
    Tokenizer.initialize(OpenAITokenizer, tokenizer_model)
    new_message = UserMessage(content=" ".join(message))
    kb = KnowledgeBase(index_name_suffix=index_name_suffix)
    kb.connect()
    context_engine =  ContextEngine(knowledge_base=kb)
    llm = OpenAILLM(llm_model_name)
    chat_engine = ChatEngine(llm=llm, context_engine=context_engine, max_prompt_tokens=max_prompt_tokens, max_generated_tokens=max_generated_tokens)
    res = chat_engine.chat([new_message], stream=True)
    click.echo(click.style("\n🤖 says:\n", fg="green"))
    for chunk in res:
        click.echo(chunk.choices[0].delta.get("content", ""), nl=False)

@cli.command()
@click.option('--host', default="0.0.0.0", help='Host')
@click.option('--port', default=8000, help='Port')
@click.option('--reload/--no-reload', default=True, help='Reload')
def start(host, port, reload):
    click.echo(f"Starting Context Engine service on {host}:{port}")
    # os.system(f"uvicorn context_engine.service.app:app --host {host} --port {port} --reload {reload}")
    start_service(host, port, reload)

if __name__ == "__main__":
    cli()


