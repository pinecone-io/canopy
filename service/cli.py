import os
import click
import time

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


@click.group(help="CLI for the Context Engine")
def cli():
    pass


@cli.command()
@click.argument(
    "index-name-suffix", nargs=1, envvar="INDEX_NAME_SUFFIX", type=str, required=True
)
@click.option("--tokenizer-model", default="gpt-3.5-turbo", help="Tokenizer model")
def new(index_name_suffix, tokenizer_model):
    click.echo(
        f"Context Engine is going to create a new index: "
        f"{INDEX_NAME_PREFIX}{index_name_suffix}"
    )
    click.confirm(click.style("Do you want to continue?", fg="red"), abort=True)
    Tokenizer.initialize(OpenAITokenizer, tokenizer_model)
    kb = KnowledgeBase(index_name_suffix=index_name_suffix)
    with spinner:
        kb.create_index()
    click.echo(click.style("Success!", fg="green"))
    os.environ["INDEX_NAME_SUFFIX"] = index_name_suffix


@cli.command()
@click.argument("data-path", type=click.Path(exists=True))
@click.option(
    "--index-name-suffix",
    default=os.environ.get("INDEX_NAME_SUFFIX"),
    help="Index name suffix",
)
@click.option("--tokenizer-model", default="gpt-3.5-turbo", help="Tokenizer model")
def upsert(index_name_suffix, data_path, tokenizer_model):
    if index_name_suffix is None:
        raise ValueError("Must provide index name suffix")
    Tokenizer.initialize(OpenAITokenizer, tokenizer_model)
    if data_path is None:
        raise ValueError("Must provide data path")
    click.echo(
        f"Context Engine is going to upsert data from {data_path} to index: "
        f"{INDEX_NAME_PREFIX}{index_name_suffix}"
    )
    kb = KnowledgeBase(index_name_suffix=index_name_suffix)
    kb.connect()
    click.echo("")
    data = pd.read_parquet(data_path)
    click.echo(tabulate(data.head(), headers="keys", tablefmt="psql", showindex=False))
    click.confirm(click.style("Does this data look right?", fg="red"), abort=True)
    kb.upsert_dataframe(data)
    click.echo(click.style("Success!", fg="green"))


def is_healthy(url: str):
    try:
        health_url = os.path.join(url, "health")
        res = requests.get(health_url)
        res.raise_for_status()
        return res.ok
    except Exception:
        return False


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
    pinecone_res = openai.ChatCompletion.create(
        model=model, messages=history, stream=stream, api_base=api_base
    )
    end = time.time()
    duration_in_sec = end - start
    click.echo(click.style(f"\n {speaker} says:\n", fg=speaker_color))
    if stream:
        for chunk in pinecone_res:
            intenal_model = chunk.model
            text = chunk.choices[0].delta.get("content", "")
            output += text
            click.echo(text, nl=False)
        click.echo()
        debug_info = ChatDebugInfo(
            intenal_model=intenal_model, duration_in_sec=round(duration_in_sec, 2)
        )
    else:
        intenal_model = pinecone_res.model
        text = pinecone_res.choices[0].message.get("content", "")
        output = text
        click.echo(text, nl=False)
        debug_info = ChatDebugInfo(
            intenal_model=intenal_model,
            duration_in_sec=duration_in_sec,
            prompt_tokens=pinecone_res.usage.prompt_tokens,
            generated_tokens=pinecone_res.usage.completion_tokens,
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
    "--with-vanilla-llm/--no-vanilla-llm",
    default=False,
    help="Direct chat with the model",
)
@click.option("--chat-service-url", default="http://0.0.0.0:8000")
@click.option(
    "--index-name-suffix",
    default=os.environ.get("INDEX_NAME_SUFFIX"),
    help="Index name suffix",
)
def chat(
    index_name_suffix, chat_service_url, with_vanilla_llm, debug_info, stream
):
    if not is_healthy(chat_service_url):
        raise ValueError(f"Context Engine service is not running at {chat_service_url}")

    history_with_pinecone = []
    history_without_pinecone = []

    while True:
        click.echo(click.style("\nUser message:\n", fg="bright_blue"), nl=True)
        message = click.get_text_stream("stdin").readline()

        dubug_info = _chat(
            speaker="🤖 + Pinecone",
            speaker_color="green",
            model="",
            history=history_with_pinecone,
            message=message,
            stream=stream,
            api_base=os.path.join(chat_service_url, "context"),
            print_debug_info=debug_info,
        )

        if with_vanilla_llm:
            _ = _chat(
                speaker="Just 🤖",
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
@click.option("--reload/--no-reload", default=False, help="Reload")
def start(host, port, reload):
    click.echo(f"Starting Context Engine service on {host}:{port}")
    start_service(host, port, reload)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host")
@click.option("--port", default=8000, help="Port")
def stop(host, port):
    import subprocess

    p1 = subprocess.Popen(["lsof", "-t", "-i", f"tcp:{port}"], stdout=subprocess.PIPE)
    running_server_id = p1.stdout.read().decode("utf-8").strip()
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
