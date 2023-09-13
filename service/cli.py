import os
import click
import time
import sys

import requests
from dotenv import load_dotenv
from tabulate import tabulate

import pandas as pd
import openai

from context_engine.knoweldge_base import KnowledgeBase
from context_engine.knoweldge_base.knowledge_base import INDEX_NAME_PREFIX
from context_engine.knoweldge_base.tokenizer import OpenAITokenizer, Tokenizer

from service.app import start as start_service
from service.cli_spinner import Spinner
from service.api_models import ChatDebugInfo


dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

spinner = Spinner()


def is_healthy(url: str):
    try:
        health_url = os.path.join(url, "health")
        res = requests.get(health_url)
        res.raise_for_status()
        return res.ok
    except Exception:
        return False


@click.group()
def cli():
    """
    CLI for Pinecone Context Engine. Actively developed by Pinecone.
    To use the CLI, you need to have a Pinecone account. Visit https://www.pinecone.io/ to sign up for free.
    """
    pass


@cli.command(help="Check if Context Engine service is running")
@click.option("--host", default="0.0.0.0", help="Host")
@click.option("--port", default=8000, help="Port")
@click.option("--ssl/--no-ssl", default=False, help="SSL")
def health(host, port, ssl):
    ssl_str = "s" if ssl else ""
    service_url = f"http{ssl_str}://{host}:{port}"
    if not is_healthy(service_url):
        msg = (
            f"Context Engine service is not running! on {service_url}"
            + " please run `context-engine start`"
        )
        click.echo(click.style(msg, fg="red"), err=True)
        sys.exit(1)
    else:
        click.echo(click.style("Context Engine service is healthy!", fg="green"))
        return


@cli.command()
@click.argument("index-name", nargs=1, envvar="INDEX_NAME", type=str, required=True)
@click.option("--tokenizer-model", default="gpt-3.5-turbo", help="Tokenizer model")
def new(index_name, tokenizer_model):
    click.echo(f"Context Engine is going to create a new index: ", nl=False)
    click.echo(click.style(f"{INDEX_NAME_PREFIX}{index_name}", fg="green"))
    click.confirm(click.style("Do you want to continue?", fg="red"), abort=True)
    Tokenizer.initialize(OpenAITokenizer, tokenizer_model)
    with spinner:
        kb = KnowledgeBase.create_with_new_index(
            index_name=index_name,
            encoder=KnowledgeBase.DEFAULT_RECORD_ENCODER(),
            chunker=KnowledgeBase.DEFAULT_CHUNKER(),
        )
    click.echo(click.style("Success!", fg="green"))
    os.environ["INDEX_NAME"] = index_name


@cli.command()
@click.argument("data-path", type=click.Path(exists=True))
@click.option(
    "--index-name",
    default=os.environ.get("INDEX_NAME"),
    help="Index name",
)
@click.option("--tokenizer-model", default="gpt-3.5-turbo", help="Tokenizer model")
def upsert(index_name, data_path, tokenizer_model):
    if index_name is None:
        msg = 'Index name is not provided, please provide it with --index-name or set it with env var `export INDEX_NAME="MY_INDEX_NAME"`'
        click.echo(click.style(msg, fg="red"), err=True)
        sys.exit(1)
    Tokenizer.initialize(OpenAITokenizer, tokenizer_model)
    if data_path is None:
        msg = "Data path is not provided, please provide it with --data-path or set it with env var"
        click.echo(click.style(msg, fg="red"), err=True)
        sys.exit(1)
    click.echo(
        f"Context Engine is going to upsert data from {data_path} to index: "
        f"{INDEX_NAME_PREFIX}{index_name}"
    )
    kb = KnowledgeBase(index_name=index_name)
    click.echo("")
    data = pd.read_parquet(data_path)
    head = data.head()
    pd.options.display.max_colwidth = 20
    pd.options.mode.chained_assignment = None
    head["text"] = head["text"].apply(lambda x: x[:50] + "...")
    head["source"] = head["source"].apply(lambda x: x[:30] + "...")
    click.echo(tabulate(head, headers="keys", tablefmt="psql", showindex=False))
    click.confirm(click.style("Does this data look right?", fg="red"), abort=True)
    kb.upsert_dataframe(data)
    click.echo(click.style("Success!", fg="green"))


def _chat(
    speaker,
    speaker_color,
    model,
    history,
    message,
    api_base=None,
    stream=True,
    print_debug_info=False,
):
    output = ""
    history += [{"role": "user", "content": message}]
    start = time.time()
    openai_response = openai.ChatCompletion.create(
        model=model, messages=history, stream=stream, api_base=api_base
    )
    end = time.time()
    duration_in_sec = end - start
    click.echo(click.style(f"\n {speaker}:\n", fg=speaker_color))
    if stream:
        for chunk in openai_response:
            intenal_model = chunk.model
            text = chunk.choices[0].delta.get("content", "")
            output += text
            click.echo(text, nl=False)
        click.echo()
        debug_info = ChatDebugInfo(
            intenal_model=intenal_model, duration_in_sec=round(duration_in_sec, 2)
        )
    else:
        intenal_model = openai_response.model
        text = openai_response.choices[0].message.get("content", "")
        output = text
        click.echo(text, nl=False)
        debug_info = ChatDebugInfo(
            intenal_model=intenal_model,
            duration_in_sec=duration_in_sec,
            prompt_tokens=openai_response.usage.prompt_tokens,
            generated_tokens=openai_response.usage.completion_tokens,
        )
    if print_debug_info:
        click.echo()
        click.echo(
            click.style(f"{debug_info.to_text()}", fg="bright_black", italic=True)
        )
    history += [{"role": "assistant", "content": output}]
    return debug_info


@cli.command()
@click.option("--stream/--no-stream", default=True, help="Stream")
@click.option("--debug-info/--no-debug-info", default=False, help="Print debug info")
@click.option(
    "--rag/--no-rag",
    default=True,
    help="Direct chat with the model",
)
@click.option("--chat-service-url", default="http://0.0.0.0:8000")
@click.option(
    "--index-name",
    default=os.environ.get("INDEX_NAME"),
    help="Index name suffix",
)
def chat(index_name, chat_service_url, rag, debug_info, stream):
    if not is_healthy(chat_service_url):
        msg = (
            f"Context Engine service is not running! on {chat_service_url}"
            + " please run `context-engine start`"
        )
        click.echo(click.style(msg, fg="red"), err=True)
        sys.exit(1)

    history_with_pinecone = []
    history_without_pinecone = []

    while True:
        click.echo(click.style("\nUser message:\n", fg="bright_blue"), nl=True)
        message = click.get_text_stream("stdin").readline()

        dubug_info = _chat(
            speaker="With Context (RAG)",
            speaker_color="green",
            model="",
            history=history_with_pinecone,
            message=message,
            stream=stream,
            api_base=os.path.join(chat_service_url, "context"),
            print_debug_info=debug_info,
        )

        if not rag:
            _ = _chat(
                speaker="Without Context (No RAG)",
                speaker_color="yellow",
                model=dubug_info.intenal_model,
                history=history_without_pinecone,
                message=message,
                stream=stream,
                print_debug_info=debug_info,
            )

        click.echo(click.style("\n.", fg="bright_black"))
        click.echo(
            click.style(
                f"| {len(history_with_pinecone) // 2}", fg="bright_black", bold=True
            ),
            nl=True,
        )
        click.echo(click.style("˙▔▔▔", fg="bright_black", bold=True), nl=False)
        click.echo(click.style("˙", fg="bright_black", bold=True))


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host")
@click.option("--port", default=8000, help="Port")
@click.option("--ssl/--no-ssl", default=False, help="SSL")
@click.option("--reload/--no-reload", default=False, help="Reload")
def start(host, port, ssl, reload):
    click.echo(f"Starting Context Engine service on {host}:{port}")
    start_service(host, port, reload)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host")
@click.option("--port", default=8000, help="Port")
@click.option("--ssl/--no-ssl", default=False, help="SSL")
def stop(host, port, ssl):
    ssl_str = "s" if ssl else ""
    service_url = f"http{ssl_str}://{host}:{port}"

    if not is_healthy(service_url):
        msg = (
            f"Context Engine service is not running! on {service_url}"
            + " please run `context-engine start`"
        )
        click.echo(click.style(msg, fg="red"), err=True)
        sys.exit(1)

    import subprocess

    p1 = subprocess.Popen(["lsof", "-t", "-i", f"tcp:{port}"], stdout=subprocess.PIPE)
    running_server_id = p1.stdout.read().decode("utf-8").strip()
    if running_server_id == "":
        click.echo(
            click.style(
                f"Did not find active process for context-engine service on {host}:{port}",
                fg="red",
            )
        )
        sys.exit(1)

    msg = (
        "Warning, this will invoke in process kill to the PID of the service, this method is not recommended!"
        + " We recommend ctrl+c on the terminal where you started the service"
        + " as this will allow the service to gracefully shutdown"
    )
    click.echo(click.style(msg, fg="yellow"))

    click.confirm(
        click.style(
            f"Stopping Context Engine service on {host}:{port} with pid "
            f"{running_server_id}",
            fg="red",
        ),
        abort=True,
    )
    p2 = subprocess.Popen(
        ["kill", "-9", running_server_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    kill_result = p2.stderr.read().decode("utf-8").strip()
    if kill_result == "":
        click.echo(click.style("Success!", fg="green"))
    else:
        click.echo(click.style(kill_result, fg="red"))
        click.echo(click.style("Failed!", fg="red"))


if __name__ == "__main__":
    cli()
