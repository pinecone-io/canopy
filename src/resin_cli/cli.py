import os
import signal
import subprocess
from typing import Dict, Any

import click
import time

import requests
import yaml
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

import pandas as pd
import openai
from openai.error import APIError as OpenAI_APIError
from urllib.parse import urljoin

from resin.knowledge_base import KnowledgeBase
from resin.models.data_models import Document
from resin.tokenizer import Tokenizer
from resin_cli.data_loader import (
    load_from_path,
    IDsNotUniqueError,
    DocumentsValidationError)
from resin_cli.errors import CLIError

from resin import __version__

from .app import start as start_service
from .cli_spinner import Spinner
from .api_models import ChatDebugInfo


load_dotenv()
if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")

spinner = Spinner()
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def check_service_health(url: str):
    try:
        res = requests.get(urljoin(url, "/health"))
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
    except RuntimeError as e:
        msg = (
            f"{str(e)}\n"
            "Credentials should be set by the PINECONE_API_KEY and PINECONE_ENVIRONMENT"
            " environment variables. "
            "Please visit https://www.pinecone.io/docs/quick-start/ for more details."
        )
        raise CLIError(msg)
    try:
        openai.Model.list()
    except Exception:
        msg = (
            "Failed to connect to OpenAI, please make sure that the OPENAI_API_KEY "
            "environment variable is set correctly.\n"
            "Please visit https://platform.openai.com/account/api-keys for more details"
        )
        raise CLIError(msg)
    click.echo("Resin: ", nl=False)
    click.echo(click.style("Ready\n", bold=True, fg="green"))


def _initialize_tokenizer():
    try:
        Tokenizer.initialize()
    except Exception as e:
        msg = f"Failed to initialize tokenizer. Reason:\n{e}"
        raise CLIError(msg)


def _load_kb_config(config_file: str) -> Dict[str, Any]:
    try:
        with open(os.path.join("config", config_file), 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        msg = f"Failed to load config file {config_file}. Reason:\n{e}"
        raise CLIError(msg)

    if "knowledge_base" in config:
        kb_config = config.get("knowledge_base", None)
    elif "chat_engine" in config:
        kb_config = config["chat_engine"]\
            .get("context_engine", {})\
            .get("knowledge_base", None)
    else:
        kb_config = None

    if kb_config is None:
        msg = (f"Did not find a `knowledge_base` configuration in {config_file}",
               "Would you like to use the default configuration?")
        click.confirm(click.style(msg, fg="red"), abort=True)
        kb_config = {}
    return kb_config


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, "-v", "--version", prog_name="Resin")
@click.pass_context
def cli(ctx):
    """
    \b
    CLI for Pinecone Resin. Actively developed by Pinecone.
    To use the CLI, you need to have a Pinecone account.
    Visit https://www.pinecone.io/ to sign up for free.
    """
    if ctx.invoked_subcommand is None:
        validate_connection()
        click.echo(ctx.get_help())


@cli.command(help="Check if resin service is running and healthy.")
@click.option("--url", default="http://0.0.0.0:8000",
              help="Resin's service url. Defaults to http://0.0.0.0:8000")
def health(url):
    check_service_health(url)
    click.echo(click.style("Resin service is healthy!", fg="green"))
    return


@cli.command(
    help=(
        """Create a new Pinecone index that that will be used by Resin.
        \b
        A Resin service can not be started without a Pinecone index which is configured to work with Resin.
        This command will create a new Pinecone index and configure it in the right schema.

        If the embedding vectors' dimension is not explicitly configured in
        the config file - the embedding model will be tapped with a single token to
        infer the dimensionality of the embedding space.
        """  # noqa: E501
    )
)
@click.argument("index-name", nargs=1, envvar="INDEX_NAME", type=str, required=True)
@click.option("--config", "-c", default="config.yaml",
              help="Name of the resin config file. The file needs to be in the"
                   "`config/` directory. Defaults to `config.yaml`")
def new(index_name, config):
    _initialize_tokenizer()
    kb_config = _load_kb_config(config)
    kb = KnowledgeBase.from_config(kb_config, index_name=index_name)
    click.echo("Resin is going to create a new index: ", nl=False)
    click.echo(click.style(f"{kb.index_name}", fg="green"))
    click.confirm(click.style("Do you want to continue?", fg="red"), abort=True)
    with spinner:
        try:
            kb.create_resin_index()
        # TODO: kb should throw a specific exception for each case
        except Exception as e:
            msg = f"Failed to create a new index. Reason:\n{e}"
            raise CLIError(msg)
    click.echo(click.style("Success!", fg="green"))
    os.environ["INDEX_NAME"] = index_name


@cli.command(
    help=(
        """
        \b
        Upload local data files containing documents to the Resin service.

        Load all the documents from data file or a directory containing multiple data files.
        The allowed formats are .jsonl and .parquet.
        """  # noqa: E501
    )
)
@click.argument("data-path", type=click.Path(exists=True))
@click.option(
    "--index-name",
    default=os.environ.get("INDEX_NAME"),
    help="The name of the index to upload the data to. "
         "Inferred from INDEX_NAME env var if not provided."
)
@click.option("--batch-size", default=10,
              help="Number of documents to upload in each batch. Defaults to 10.")
@click.option("--stop-on-error/--no-stop-on-error", default=True,
              help="Whether to stop when the first error arises. Defaults to True.")
@click.option("--config", "-c", default="config.yaml",
              help="Name of the resin config file. The file needs to be in the"
                   "`config/` directory. Defaults to `config.yaml`")
def upsert(index_name: str,
           data_path: str,
           batch_size: int,
           stop_on_error: bool,
           config: str):
    if index_name is None:
        msg = (
            "No index name provided. Please set --index-name or INDEX_NAME environment "
            "variable."
        )
        raise CLIError(msg)

    _initialize_tokenizer()

    kb_config = _load_kb_config(config)
    kb = KnowledgeBase.from_config(kb_config, index_name=index_name)
    try:
        kb.connect()
    except RuntimeError as e:
        # TODO: kb should throw a specific exception for each case
        msg = str(e)
        if "credentials" in msg:
            msg += ("\nCredentials should be set by the PINECONE_API_KEY and "
                    "PINECONE_ENVIRONMENT environment variables. Please visit "
                    "https://www.pinecone.io/docs/quick-start/ for more details.")
        raise CLIError(msg)

    click.echo("Resin is going to upsert data from ", nl=False)
    click.echo(click.style(f"{data_path}", fg="yellow"), nl=False)
    click.echo(" to index: ")
    click.echo(click.style(f'{kb.index_name} \n', fg='green'))
    with spinner:
        try:
            data = load_from_path(data_path)
        except IDsNotUniqueError:
            msg = (
                "The data contains duplicate IDs, please make sure that each document"
                " has a unique ID, otherwise documents with the same ID will overwrite"
                " each other"
            )
            raise CLIError(msg)
        except DocumentsValidationError:
            msg = (
                f"One or more rows have failed data validation. The rows in the"
                f"data file should be in the schema: {Document.__annotations__}."
            )
            raise CLIError(msg)
        except Exception:
            msg = (
                f"A unexpected error while loading the data from files in {data_path}. "
                "Please make sure the data is in valid `jsonl` or `parquet` format."
            )
            raise CLIError(msg)
        pd.options.display.max_colwidth = 20
    click.echo(pd.DataFrame([doc.dict(exclude_none=True) for doc in data[:5]]))
    click.echo(click.style(f"\nTotal records: {len(data)}"))
    click.confirm(click.style("\nDoes this data look right?", fg="red"),
                  abort=True)

    pbar = tqdm(total=len(data), desc="Upserting documents")
    failed_docs = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        try:
            kb.upsert(data)
        except Exception as e:
            if stop_on_error or len(failed_docs) > len(data) // 10:
                msg = (
                    f"Failed to upsert data to index {kb.index_name}. "
                    f"Underlying error: {e}"
                )
                raise CLIError(msg)
            else:
                failed_docs.extend([_.id for _ in batch])

        pbar.update(len(batch))

    if failed_docs:
        msg = (
            f"Failed to upsert the following documents to index {kb.index_name}: "
            f"{failed_docs}"
        )
        raise CLIError(msg)

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
    except (Exception, OpenAI_APIError) as e:
        err = e.http_body if isinstance(e, OpenAI_APIError) else str(e)
        msg = f"Oops... something went wrong. The error I got is: {err}"
        raise CLIError(msg)
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
        """
        Debugging tool for chatting with the Resin RAG service.

        Run an interactive chat with the Resin RAG service, for debugging and demo
        purposes. A prompt is provided for the user to enter a message, and the
        RAG-infused ChatBot will respond. You can continue the conversation by entering
        more messages. Hit Ctrl+C to exit.

        To compare RAG-infused ChatBot with the original LLM, run with the `--baseline`
        flag, which would display both models' responses side by side.
        """

    )
)
@click.option("--stream/--no-stream", default=True,
              help="Stream the response from the RAG chatbot word by word")
@click.option("--debug/--no-debug", default=False,
              help="Print additional debugging information")
@click.option("--baseline/--no-baseline", default=False,
              help="Compare RAG-infused Chatbot with baseline LLM",)
@click.option("--chat-service-url", default="http://0.0.0.0:8000",
              help="URL of the Resin service to use. Defaults to http://0.0.0.0:8000")
def chat(chat_service_url, baseline, debug, stream):
    check_service_health(chat_service_url)
    note_msg = (
        "🚨 Note 🚨\n"
        "Chat is a debugging tool, it is not meant to be used for production!"
    )
    for c in note_msg:
        click.echo(click.style(c, fg="red"), nl=False)
        time.sleep(0.01)
    click.echo()
    note_white_message = (
        "This method should be used by developers to test the RAG data and model"
        "during development. "
        "When you are ready to deploy, run the Resin service as a REST API "
        "backend for your chatbot UI. \n\n"
        "Let's Chat!"
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

        if baseline:
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
        """
        \b
        Start the Resin service.
        This command will launch a uvicorn server that will serve the Resin API.

        If you like to try out the chatbot, run `resin chat` in a separate terminal
        window.
        """
    )
)
@click.option("--host", default="0.0.0.0",
              help="Hostname or ip address to bind the server to. Defaults to 0.0.0.0")
@click.option("--port", default=8000,
              help="TCP port to bind the server to. Defaults to 8000")
@click.option("--reload/--no-reload", default=False,
              help="Set the server to reload on code changes. Defaults to False")
@click.option("--config", "-c", default="config.yaml",
              help="Name of the resin config file. The file needs to be in the"
                   "`config/` directory. Defaults to `config.yaml`")
def start(host, port, reload, config):
    note_msg = (
        "🚨 Note 🚨\n"
        "For debugging only. To run the Resin service in production, run the command:\n"
        "gunicorn resin_cli.app:app --worker-class uvicorn.workers.UvicornWorker "
        f"--bind {host}:{port} --workers <num_workers>"
    )
    for c in note_msg:
        click.echo(click.style(c, fg="red"), nl=False)
        time.sleep(0.01)
    click.echo()

    click.echo(f"Starting Resin service on {host}:{port}")
    config_file = os.path.join("config", config)
    start_service(host, port=port, reload=reload, config_file=config_file)


@cli.command(
    help=(
        """
        \b
        Stop the Resin service.
        This command will send a shutdown request to the Resin service.
        """
    )
)
@click.option("url", "--url", default="http://0.0.0.0:8000",
              help="URL of the Resin service to use. Defaults to http://0.0.0.0:8000")
def stop(url):
    # Check if the service was started using Gunicorn
    res = subprocess.run(["pgrep", "-f", "gunicorn resin_cli.app:app"],
                         capture_output=True)
    output = res.stdout.decode("utf-8").split()

    # If Gunicorn was used, kill all Gunicorn processes
    if output:
        msg = ("It seems that Resin service was launched using Gunicorn.\n"
               "Do you want to kill all Gunicorn processes?")
        click.confirm(click.style(msg, fg="red"), abort=True)
        try:
            subprocess.run(["pkill", "-f", "gunicorn resin_cli.app:app"], check=True)
        except subprocess.CalledProcessError:
            try:
                [os.kill(int(pid), signal.SIGINT) for pid in output]
            except OSError:
                msg = (
                    "Could not kill Gunicorn processes. Please kill them manually."
                    f"Found process ids: {output}"
                )
                raise CLIError(msg)

    try:
        res = requests.get(urljoin(url, "/shutdown"))
        res.raise_for_status()
        return res.ok
    except requests.exceptions.ConnectionError:
        msg = f"""
        Could not find Resin service on {url}.
        """
        raise CLIError(msg)


if __name__ == "__main__":
    cli()
