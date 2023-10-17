import os

import click
import time
import sys
import subprocess

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

import pandas as pd
import openai

from resin.knoweldge_base import KnowledgeBase
from resin.models.data_models import Document
from resin.tokenizer import OpenAITokenizer, Tokenizer
from resin_cli.data_loader import (
    load_from_path,
    CLIError,
    IDsNotUniqueError,
    DocumentsValidationError)

from resin import __version__

from .app import start as start_service
from .cli_spinner import Spinner
from .api_models import ChatDebugInfo


dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

spinner = Spinner()


def check_service_health(url: str):
    try:
        health_url = os.path.join(url, "health")
        res = requests.get(health_url)
        res.raise_for_status()
        return res.ok
    except requests.exceptions.ConnectionError:
        msg = f"""
        Resin service is not running on {url}. 
        please run `resin start`
        """
        raise CLIError(msg)

    except requests.exceptions.HTTPError as e:
        error = e.response.json().get("detail", None) or e.response.text
        msg = (
            f"Resin service on {url} is not healthy, failed with error: {error}"
        )
        raise CLIError(msg)

@retry(wait=wait_fixed(5), stop=stop_after_attempt(6))
def wait_for_service(chat_service_url: str):
    check_service_health(chat_service_url)


def validate_connection():
    try:
        KnowledgeBase._connect_pinecone()
    except Exception:
        msg = (
            "Error: Failed to connect to Pinecone index, please make sure"
            + " you have set the right env vars"
            + " PINECONE_API_KEY, INDEX_NAME, PINECONE_ENVIRONMENT"
        )
        click.echo(click.style(msg, fg="red"), err=True)
        sys.exit(1)
    try:
        openai.Model.list()
    except Exception:
        msg = (
            "Error: Failed to connect to OpenAI, please make sure"
            + " you have set the right env vars"
        )
        click.echo(click.style(msg, fg="red"), err=True)
        sys.exit(1)
    click.echo("Resin: ", nl=False)
    click.echo(click.style("Ready\n", bold=True, fg="green"))


@click.group(invoke_without_command=True)
@click.version_option(__version__, "-v", "--version", prog_name="Resin")
@click.pass_context
def cli(ctx):
    """
    CLI for Pinecone Resin. Actively developed by Pinecone.
    To use the CLI, you need to have a Pinecone account.
    Visit https://www.pinecone.io/ to sign up for free.
    """
    if ctx.invoked_subcommand is None:
        validate_connection()
        click.echo(ctx.get_help())


@cli.command(help="Check if resin service is running by sending a health check request")
@click.option("--host", default="0.0.0.0", help="Resin's service hostname")
@click.option("--port", default=8000, help="The port of the resin service")
@click.option("--ssl/--no-ssl", default=False, help="Whether to use ssl for the "
                                                    "connection to resin service")
def health(host, port, ssl):
    ssl_str = "s" if ssl else ""
    service_url = f"http{ssl_str}://{host}:{port}"
    check_service_health(service_url)
    click.echo(click.style("Resin service is healthy!", fg="green"))
    return


@cli.command(
    help=(
        "New command sets up a new index in Pinecone that is configured for Resin."
        + " This will automatically tap the embedding model with a single toeken to "
        + "assert for the dimensionality of the embedding space. This will also set up "
        + "the index with the right schema for Resin."
    )
)
@click.argument("index-name", nargs=1, envvar="INDEX_NAME", type=str, required=True)
@click.option("--tokenizer-model", default="gpt-3.5-turbo", help="Tokenizer model")
def new(index_name, tokenizer_model):
    kb = KnowledgeBase(index_name=index_name)
    click.echo("Resin is going to create a new index: ", nl=False)
    click.echo(click.style(f"{kb.index_name}", fg="green"))
    click.confirm(click.style("Do you want to continue?", fg="red"), abort=True)
    Tokenizer.initialize(OpenAITokenizer, tokenizer_model)
    with spinner:
        try:
            kb.create_resin_index()
        # TODO: kb should throw a specific exception for each case
        except Exception as e:
            msg = "Error: Failed to create a new index"
            click.echo(click.style(msg, fg="red"), err=True, nl=False)
            click.echo(f" Reason: {e}")
            sys.exit(1)
    click.echo(click.style("Success!", fg="green"))
    os.environ["INDEX_NAME"] = index_name


@cli.command(
    help=(
        "Upsert allows you to load a loacal data file into a your Resin index."
        + " The allowed formats are .jsonl and .parquet. The data will be validated"
    )
)
@click.argument("data-path", type=click.Path(exists=True))
@click.option(
    "--index-name",
    default=os.environ.get("INDEX_NAME"),
    help="Index name",
)
@click.option("--tokenizer-model", default="gpt-3.5-turbo", help="Tokenizer model")
def upsert(index_name, data_path, tokenizer_model):
    if index_name is None:
        msg = ("Index name is not provided, please provide it with" +
               ' --index-name or set it with env var + '
               '`export INDEX_NAME="MY_INDEX_NAME`')
        click.echo(click.style(msg, fg="red"), err=True)
        sys.exit(1)
    try:
        Tokenizer.initialize(OpenAITokenizer, tokenizer_model)
    except Exception:
        msg = "Error: Failed to initialize tokenizer"
        click.echo(click.style(msg, fg="red"), err=True)
        sys.exit(1)
    if data_path is None:
        msg = ("Data path is not provided,"
               " please provide it with --data-path or set it with env var")
        click.echo(click.style(msg, fg="red"), err=True)
        sys.exit(1)

    kb = KnowledgeBase(index_name=index_name)
    try:
        kb.connect()
    except Exception:
        msg = (
            "Error: Failed to connect to Pinecone index, please make sure"
            " you have set the right env vars"
            " PINECONE_API_KEY, INDEX_NAME, PINECONE_ENVIRONMENT"
        )
        click.echo(click.style(msg, fg="red"), err=True)
        sys.exit(1)

    click.echo("Resin is going to upsert data from ", nl=False)
    click.echo(click.style(f"{data_path}", fg="yellow"), nl=False)
    click.echo(" to index: ")
    click.echo(click.style(f'{kb.index_name} \n', fg='green'))
    with spinner:
        try:
            data = load_from_path(data_path)
        except IDsNotUniqueError:
            msg = (
                "Error: the id field on the data is not unique"
                + " this will cause records to override each other on upsert"
                + " please make sure the id field is unique"
            )
            click.echo(click.style(msg, fg="red"), err=True)
            sys.exit(1)
        except DocumentsValidationError:
            msg = (
                "Error: one or more rows have not passed validation"
                + " data should agree with the Document Schema"
                + f" on {Document.__annotations__}"
                + " please make sure the data is valid"
            )
            click.echo(click.style(msg, fg="red"), err=True)
            sys.exit(1)
        except Exception:
            msg = (
                "Error: an unexpected error has occured in loading data from files"
                + " it may be due to issue with the data format"
                + " please make sure the data is valid, and can load with pandas"
            )
            click.echo(click.style(msg, fg="red"), err=True)
            sys.exit(1)
        pd.options.display.max_colwidth = 20
    click.echo(data[0].json(exclude_none=True, indent=2))
    click.confirm(click.style("\nDoes this data look right?", fg="red"), abort=True)
    try:
        kb.upsert(data)
    except Exception:
        msg = (
            "Error: Failed to upsert data to index"
            f" {kb.index_name}"
            " this could be due to connection issues"
            " please re-run `resin upsert`"
        )
        click.echo(click.style(msg, fg="red"), err=True)
        sys.exit(1)
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
    try:
        openai_response = openai.ChatCompletion.create(
            model=model, messages=history, stream=stream, api_base=api_base
        )
    except Exception as e:
        msg = "Oops... something went wrong with the LLM" + " the error I got is: "
        click.echo(click.style(msg, fg="red"), err=True, nl=False)
        click.echo(f"{e}")
        sys.exit(1)
    end = time.time()
    duration_in_sec = end - start
    click.echo(click.style(f"\n> AI {speaker}:\n", fg=speaker_color))
    if stream:
        for chunk in openai_response:
            openai_response_id = chunk.id
            intenal_model = chunk.model
            text = chunk.choices[0].delta.get("content", "")
            output += text
            click.echo(text, nl=False)
        click.echo()
        debug_info = ChatDebugInfo(
            id=openai_response_id,
            intenal_model=intenal_model,
            duration_in_sec=round(duration_in_sec, 2),
        )
    else:
        intenal_model = openai_response.model
        text = openai_response.choices[0].message.get("content", "")
        output = text
        click.echo(text, nl=False)
        debug_info = ChatDebugInfo(
            id=openai_response.id,
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


@cli.command(
    help=(
        "Chat allows you to chat with your Resin index, "
        + "Chat is a debugging tool, it is not meant to be used for production"
    )
)
@click.option("--stream/--no-stream", default=True, help="Stream")
@click.option("--debug/--no-debug", default=False, help="Print debug info")
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
def chat(index_name, chat_service_url, rag, debug, stream):
    check_service_health(chat_service_url)
    note_msg = (
        "🚨 Note 🚨\n"
        + "Chat is a debugging tool, it is not meant to be used for production!"
    )
    for c in note_msg:
        click.echo(click.style(c, fg="red"), nl=False)
        time.sleep(0.01)
    click.echo()
    note_white_message = (
        "This method should be used by developers to test the model and the data"
        + " in development time, in local environment. "
        + "For production use cases, we recommend using the"
        + " Resin Service or Resin Library directly \n\n"
        + "Let's Chat!"
    )
    for c in note_white_message:
        click.echo(click.style(c, fg="white"), nl=False)
        time.sleep(0.01)
    click.echo()

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
            print_debug_info=debug,
        )

        if not rag:
            _ = _chat(
                speaker="Without Context (No RAG)",
                speaker_color="yellow",
                model=dubug_info.intenal_model,
                history=history_without_pinecone,
                message=message,
                stream=stream,
                print_debug_info=debug,
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


@cli.command(
    help=(
        "Start the Resin service, this will start a uvicorn server"
        + " that will serve the Resin API."
        + " If you are are locally debugging, you can use the --debug flag"
        + " to start a new terminal window with the right env vars for `resin chat`"
    )
)
@click.option(
    "--chat", is_flag=True, help="open a new terminal window for debugging"
)
@click.option("--host", default="0.0.0.0", help="Host")
@click.option("--port", default=8000, help="Port")
@click.option("--ssl/--no-ssl", default=False, help="SSL")
@click.option("--reload/--no-reload", default=False, help="Reload")
def start(chat, host, port, ssl, reload):
    if chat:
        command_to_run = "clear && echo Welcome to Pinecone Canopy,"
        + " run *resin chat* to start chatting with your index"

        script = f'''
        tell application "Terminal"
            activate
            do script "{command_to_run}"
        end tell
        '''

        subprocess.run(["osascript", "-e", script], env=os.environ.copy())
    click.echo(f"Starting Resin service on {host}:{port}")
    start_service(host, port, reload)


@cli.command(
    help=(
        "Stop the Resin service, this will kill the uvicorn server"
        + " that is serving the Resin API."
        + " This method is not recommended,"
        + " as it will kill the server by looking for the PID"
        + " of the server, instead, we recommend using"
        + " ctrl+c on the terminal where you started"
    )
)
@click.option("--host", default="0.0.0.0", help="Host")
@click.option("--port", default=8000, help="Port")
@click.option("--ssl/--no-ssl", default=False, help="SSL")
def stop(host, port, ssl):
    ssl_str = "s" if ssl else ""
    service_url = f"http{ssl_str}://{host}:{port}"

    check_service_health(service_url)

    import subprocess

    p1 = subprocess.Popen(["lsof", "-t", "-i", f"tcp:{port}"], stdout=subprocess.PIPE)
    running_server_id = p1.stdout.read().decode("utf-8").strip()
    if running_server_id == "":
        click.echo(
            click.style(
                "Did not find active process for Resin service" + f" on {host}:{port}",
                fg="red",
            )
        )
        sys.exit(1)

    msg = (
        "Warning, this will invoke in process kill"
        + " to the PID of the service, this method is not recommended!"
        + " We recommend ctrl+c on the terminal where you started the service"
        + " as this will allow the service to gracefully shutdown"
    )
    click.echo(click.style(msg, fg="yellow"))

    click.confirm(
        click.style(
            f"Stopping Resin service on {host}:{port} with pid " f"{running_server_id}",
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
