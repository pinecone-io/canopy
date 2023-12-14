import os
import signal
import subprocess
from typing import Dict, Any, Optional, List, Iterable

import click
from prompt_toolkit import prompt
import time

import requests
import yaml
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

import pandas as pd
import openai
from openai import APIError as OpenAI_APIError
from urllib.parse import urljoin

from canopy.knowledge_base import KnowledgeBase
from canopy.knowledge_base import connect_to_pinecone
from canopy.knowledge_base.chunker import Chunker
from canopy.chat_engine import ChatEngine
from canopy.models.data_models import Document, UserMessage
from canopy.tokenizer import Tokenizer
from canopy_cli.data_loader import (
    load_from_path,
    IDsNotUniqueError,
    DocumentsValidationError)
from canopy_cli.errors import CLIError

from canopy import __version__

from canopy_server.app import start as start_server, API_VERSION
from .cli_spinner import Spinner
from canopy_server.models.v1.api_models import ChatDebugInfo


load_dotenv()


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
DEFAULT_SERVER_URL = f"http://localhost:8000/{API_VERSION}"
spinner = Spinner()


def check_server_health(url: str):
    try:
        res = requests.get(urljoin(url, "/health"))
        res.raise_for_status()
        return res.ok
    except requests.exceptions.ConnectionError:
        msg = f"""
        Canopy server is not running on {url}.
        please run `canopy start`
        """
        raise CLIError(msg)

    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            error = e.response.json().get("detail", None) or e.response.text
        else:
            error = str(e)
        msg = (
            f"Canopy server on {url} is not healthy, failed with error: {error}"
        )
        raise CLIError(msg)


@retry(reraise=True, wait=wait_fixed(5), stop=stop_after_attempt(6))
def wait_for_server(chat_server_url: str):
    check_server_health(chat_server_url)


def validate_pinecone_connection():
    try:
        connect_to_pinecone()
    except RuntimeError as e:
        msg = (
            f"{str(e)}\n"
            "Credentials should be set by the PINECONE_API_KEY and PINECONE_ENVIRONMENT"
            " environment variables.\n"
            "Please visit https://www.pinecone.io/docs/quickstart/ for more details."
        )
        raise CLIError(msg)


def _initialize_tokenizer():
    try:
        Tokenizer.initialize()
    except Exception as e:
        msg = f"Failed to initialize tokenizer. Reason:\n{e}"
        raise CLIError(msg)


def _read_config_file(config_file: Optional[str]) -> Dict[str, Any]:
    if config_file is None:
        return {}

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        msg = f"Failed to load config file {config_file}. Reason:\n{e}"
        raise CLIError(msg)

    return config


def _load_kb_config(config_file: Optional[str]) -> Dict[str, Any]:
    config = _read_config_file(config_file)
    if not config:
        return {}

    if "knowledge_base" in config:
        kb_config = config.get("knowledge_base", None)
    elif "chat_engine" in config:
        kb_config = config["chat_engine"]\
            .get("context_engine", {})\
            .get("knowledge_base", None)
    else:
        kb_config = None

    if kb_config is None:
        msg = (f"Did not find a `knowledge_base` configuration in {config_file}, "
               "Would you like to use the default configuration?")
        click.confirm(click.style(msg, fg="red"), abort=True)
        kb_config = {}
    return kb_config


def _validate_chat_engine(config_file: Optional[str]):
    config = _read_config_file(config_file)
    Tokenizer.initialize()
    try:
        # If the server itself will fail, we can't except the error, since it's running
        # in a different process. Try to load and run the ChatEngine so we can catch
        # any errors and print a nice message.
        chat_engine = ChatEngine.from_config(config.get("chat_engine", {}))
        chat_engine.max_generated_tokens = 5
        chat_engine.context_engine.knowledge_base.connect()
        chat_engine.chat(
            [UserMessage(content="This is a health check. Are you alive? Be concise")]
        )
    except Exception as e:
        msg = f"Failed to initialize Canopy server. Reason:\n{e}"
        if config_file:
            msg += f"\nPlease check the configuration file {config_file}"
        raise CLIError(msg)
    finally:
        Tokenizer.clear()


class CanopyCommandGroup(click.Group):
    """
    A custom click Group that lets us control the order of commands in the help menu.
    """
    def __init__(self, name=None, commands=None, **attrs):
        super().__init__(name, commands, **attrs)
        self._commands_order = {
            "new": 0,
            "upsert": 1,
            "start": 2,
            "chat": 3,
            "health": 4,
            "stop": 5,
            "api-docs": 6,

        }

    def list_commands(self, ctx):
        return sorted(self.commands, key=lambda x: self._commands_order.get(x, 1000))


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS,
             cls=CanopyCommandGroup)
@click.version_option(__version__, "-v", "--version", prog_name="Canopy")
@click.pass_context
def cli(ctx):
    """
    \b
    CLI for Pinecone Canopy. Actively developed by Pinecone.
    To use the CLI, you need to have a Pinecone account.
    Visit https://www.pinecone.io/ to sign up for free.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command(help="Check if the Canopy server is running and healthy.")
@click.option("--url", default=DEFAULT_SERVER_URL,
              help=("Canopy's server url. "
                    f"Defaults to {DEFAULT_SERVER_URL}"))
def health(url):
    check_server_health(url)
    click.echo(click.style("Canopy server is healthy!", fg="green"))
    return


@cli.command(
    help=(
        """
        \b
        Create a new Pinecone index that will be used by Canopy.

        A Canopy service cannot be started without a Pinecone index that is configured
        to work with Canopy. This command creates a new Pinecone index and configures
        it in the right schema.

        If the embedding vectors' dimension is not explicitly configured in
        the config file, the embedding model will be tapped with a single token to
        infer the dimensionality of the embedding space.
        """  # noqa: E501
    )
)
@click.argument("index-name", nargs=1, envvar="INDEX_NAME", type=str, required=True)
@click.option("--config", "-c", default=None, envvar="CANOPY_CONFIG_FILE",
              help="Path to a canopy config file. Can also be set by the "
                   "`CANOPY_CONFIG_FILE` envrionment variable. Otherwise, the built-in"
                   "defualt configuration will be used.")
def new(index_name: str, config: Optional[str]):
    _initialize_tokenizer()
    kb_config = _load_kb_config(config)
    kb = KnowledgeBase.from_config(kb_config, index_name=index_name)
    click.echo("Canopy is going to create a new index: ", nl=False)
    click.echo(click.style(f"{kb.index_name}", fg="green"))
    click.confirm(click.style("Do you want to continue?", fg="red"), abort=True)
    with spinner:
        try:
            kb.create_canopy_index()
        # TODO: kb should throw a specific exception for failure
        except Exception as e:
            already_exists_str = f"Index {kb.index_name} already exists"
            if isinstance(e, RuntimeError) and already_exists_str in str(e):
                msg = (f"{already_exists_str}, please use a different name."
                       f"If you wish to delete the index, log in to Pinecone's "
                       f"Console: https://app.pinecone.io/")
            else:
                msg = f"Failed to create a new index. Reason:\n{e}"
            raise CLIError(msg)
    click.echo(click.style("Success!", fg="green"))
    os.environ["INDEX_NAME"] = index_name


def _batch_documents_by_chunks(chunker: Chunker,
                               documents: List[Document],
                               batch_size: int = 400) -> Iterable[List[Document]]:
    """
    Note: this is a temporary solution until we improve the upsert pipeline.
          using the chunker directly is not recommended, especially since the knowledge base also going to use it internally on the same documents.
    """  # noqa: E501
    num_chunks_in_batch = 0
    batch: List[Document] = []
    for doc in documents:
        cur_num_chunks = len(chunker.chunk_single_document(doc))
        if num_chunks_in_batch + cur_num_chunks >= batch_size:
            yield batch
            batch = []
            num_chunks_in_batch = 0
        num_chunks_in_batch += cur_num_chunks
        batch.append(doc)
    if batch:
        yield batch


@cli.command(
    help=(
        """
        \b
        Upload local data files to the Canopy service.

        Load all the documents from a data file or a directory containing multiple data
        files. The allowed formats are .jsonl, .parquet, .csv, and .txt.
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
@click.option("--allow-failures/--dont-allow-failures", default=False,
              help="On default, the upsert process will stop if any document fails to "
                   "be uploaded. "
                   "When set to True, the upsert process will continue on failure, as "
                   "long as less than 10% of the documents have failed to be uploaded.")
@click.option("--config", "-c", default=None, envvar="CANOPY_CONFIG_FILE",
              help="Path to a canopy config file. Can also be set by the "
                   "`CANOPY_CONFIG_FILE` envrionment variable. Otherwise, the built-in"
                   "defualt configuration will be used.")
def upsert(index_name: str,
           data_path: str,
           allow_failures: bool,
           config: Optional[str]):
    if index_name is None:
        msg = (
            "No index name provided. Please set --index-name or INDEX_NAME environment "
            "variable."
        )
        raise CLIError(msg)

    validate_pinecone_connection()

    _initialize_tokenizer()

    kb_config = _load_kb_config(config)
    try:
        kb = KnowledgeBase.from_config(kb_config, index_name=index_name)
    except Exception as e:
        raise CLIError(str(e))

    try:
        kb.connect()
    except RuntimeError as e:
        # TODO: kb should throw a specific exception for each case
        msg = str(e)
        raise CLIError(msg)

    click.echo("Canopy is going to upsert data from ", nl=False)
    click.echo(click.style(f'{data_path}', fg='yellow'), nl=False)
    click.echo(" to index: ")
    click.echo(click.style(f'{kb.index_name} \n', fg='green'))
    with spinner:
        try:
            data = load_from_path(data_path)
        except IDsNotUniqueError:
            msg = (
                "The data contains duplicate IDs. Please make sure that each document"
                " has a unique ID; otherwise, documents with the same ID will overwrite"
                " each other."
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
                "Please make sure the data is in valid `jsonl`, `parquet`, or `csv`"
                " format, or plaintext `.txt` files."
            )
            raise CLIError(msg)
        pd.options.display.max_colwidth = 20
    click.echo(pd.DataFrame([doc.dict(exclude_none=True) for doc in data[:5]]))
    click.echo(click.style(f"\nTotal records: {len(data)}"))
    click.confirm(click.style("\nDoes this data look right?", fg="red"),
                  abort=True)

    pbar = tqdm(total=len(data), desc="Upserting documents")
    failed_docs: List[str] = []
    first_error: Optional[str] = None
    for batch in _batch_documents_by_chunks(kb._chunker, data,
                                            batch_size=kb._encoder.batch_size):
        try:
            kb.upsert(batch)
        except Exception as e:
            if allow_failures and len(failed_docs) < len(data) // 10:
                failed_docs.extend([_.id for _ in batch])
                if first_error is None:
                    first_error = str(e)
            else:
                msg = (
                    f"Failed to upsert data to index {kb.index_name}. "
                    f"Underlying error: {e}\n"
                    f"You can allow partial failures by setting --allow-failures. "
                )
                raise CLIError(msg)

        pbar.update(len(batch))
    pbar.close()

    if failed_docs:
        msg = (
            f"Failed to upsert the following documents to index {kb.index_name}: "
            f"{failed_docs}. The first encountered error was: {first_error}"
        )
        raise CLIError(msg)

    click.echo(click.style("Success!", fg="green"))


def _chat(
    speaker,
    speaker_color,
    model,
    history,
    message,
    openai_api_key=None,
    api_base=None,
    stream=True,
    print_debug_info=False,
):
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key is None and api_base is None:
        raise CLIError(
            "No OpenAI API key provided. When using the `--no-rag` flag "
            "You will need to have a valid OpenAI API key. "
            "Please set the OPENAI_API_KEY environment "
            "variable."
        )
    output = ""
    history += [{"role": "user", "content": message}]
    client = openai.OpenAI(base_url=api_base, api_key=openai_api_key)

    start = time.time()
    try:
        openai_response = client.chat.completions.create(
            model=model, messages=history, stream=stream
        )
    except (Exception, OpenAI_APIError) as e:
        err = e.http_body if isinstance(e, OpenAI_APIError) else str(e)
        msg = f"Oops... something went wrong. The error I got is: {err}"
        raise CLIError(msg)
    end = time.time()
    duration_in_sec = end - start
    click.echo(click.style(f"\n {speaker}:\n", fg=speaker_color))
    if stream:
        for chunk in openai_response:
            openai_response_id = chunk.id
            internal_model = chunk.model
            text = chunk.choices[0].delta.content or ""
            output += text
            click.echo(text, nl=False)
        click.echo()
        debug_info = ChatDebugInfo(
            id=openai_response_id,
            internal_model=internal_model,
            duration_in_sec=round(duration_in_sec, 2),
        )
    else:
        internal_model = openai_response.model
        text = openai_response.choices[0].message.content or ""
        output = text
        click.echo(text, nl=False)
        debug_info = ChatDebugInfo(
            id=openai_response.id,
            internal_model=internal_model,
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
        Debugging tool for chatting with the Canopy RAG service.

        Run an interactive chat with the Canopy RAG service, for debugging and demo
        purposes. A prompt is provided for the user to enter a message, and the
        RAG-infused ChatBot will respond. You can continue the conversation by entering
        more messages. Hit Ctrl+C to exit.

        To compare RAG-infused ChatBot with the original LLM, run with the `--no-rag`
        flag, which would display both models' responses side by side.
        """

    )
)
@click.option("--stream/--no-stream", default=True,
              help="Stream the response from the RAG chatbot word by word.")
@click.option("--debug/--no-debug", default=False,
              help="Print additional debugging information.")
@click.option("--rag/--no-rag", default=True,
              help="Compare RAG-infused Chatbot with vanilla LLM.",)
@click.option("--chat-server-url", default=DEFAULT_SERVER_URL,
              help=("URL of the Canopy server to use."
                    f" Defaults to {DEFAULT_SERVER_URL}"))
def chat(chat_server_url, rag, debug, stream):
    check_server_health(chat_server_url)
    note_msg = (
        "🚨 Note 🚨\n"
        "Chat is a debugging tool; it is not meant to be used for production!"
    )
    for c in note_msg:
        click.echo(click.style(c, fg="red"), nl=False)
        if (stream):
            time.sleep(0.01)
    click.echo()
    note_white_message = (
        "This method should be used by developers to test the RAG data and model "
        "during development. "
        "When you are ready to deploy, run the Canopy server as a REST API "
        "backend for your chatbot UI. \n\n"
        "Let's Chat!"
    )
    for c in note_white_message:
        click.echo(click.style(c, fg="white"), nl=False)
        if (stream):
            time.sleep(0.01)
    click.echo()

    history_with_pinecone = []
    history_without_pinecone = []

    while True:
        click.echo(click.style("\nUser message: ", fg="bright_blue"), nl=False)
        click.echo(
            click.style("([Esc] followed by [Enter] to accept input)\n",
                        italic=True,
                        fg="bright_black"),
            nl=True
        )
        message = prompt("  ", multiline=True, )

        if not message:
            click.echo(click.style("Please enter a message", fg="red"))
            continue

        if message == "exit":
            click.echo(click.style("Bye!", fg="red"))
            break

        if message.isspace() or message == "":
            click.echo(click.style("Please enter a message", fg="red"))
            continue

        dubug_info = _chat(
            speaker="With Context (RAG)",
            speaker_color="green",
            model="",
            history=history_with_pinecone,
            message=message,
            stream=stream,
            openai_api_key="canopy",
            api_base=chat_server_url,
            print_debug_info=debug,
        )

        if not rag:
            _ = _chat(
                speaker="Without Context (No RAG)",
                speaker_color="yellow",
                model=dubug_info.internal_model,
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
        Start the Canopy server.

        This command launches a Uvicorn server to serve the Canopy API.

        If you would like to try out the chatbot, run `canopy chat` in a separate
        terminal window.
        """
    )
)
@click.option("--stream/--no-stream", default=True,
              help="Stream the response from the RAG chatbot word by word.")
@click.option("--host", default="0.0.0.0",
              help="Hostname or address to bind the server to. Defaults to 0.0.0.0")
@click.option("--port", default=8000,
              help="TCP port to bind the server to. Defaults to 8000")
@click.option("--reload/--no-reload", default=False,
              help="Set the server to reload on code changes. Defaults to False")
@click.option("--config", "-c", default=None, envvar="CANOPY_CONFIG_FILE",
              help="Path to a canopy config file. Can also be set by the "
                   "`CANOPY_CONFIG_FILE` envrionment variable. Otherwise, the built-in"
                   "defualt configuration will be used.")
@click.option("--index-name", default=None,
              help="Index name, if not provided already as an environment variable.")
def start(host: str, port: str, reload: bool, stream: bool,
          config: Optional[str], index_name: Optional[str]):
    validate_pinecone_connection()
    _validate_chat_engine(config)

    note_msg = (
        "🚨 Note 🚨\n"
        "For debugging only. To run the Canopy server in production, "
    )
    msg_suffix = (
        "run the command:"
        "\n"
        "gunicorn canopy_server.app:app --worker-class uvicorn.workers.UvicornWorker "
        f"--bind {host}:{port} --workers <num_workers>"
    ) if os.name != "nt" else (
        # TODO: Replace with proper instructions once we have a Dockerfile
        "please use Docker with a Gunicorn server."
    )
    for c in note_msg + msg_suffix:
        click.echo(click.style(c, fg="red"), nl=False)
        if (stream):
            time.sleep(0.01)
    click.echo()

    if index_name:
        env_index_name = os.getenv("INDEX_NAME")
        if env_index_name and index_name != env_index_name:
            raise CLIError(
                f"Index name provided via --index-name '{index_name}' does not match "
                f"the index name provided via the INDEX_NAME environment variable "
                f"'{env_index_name}'"
            )
        os.environ["INDEX_NAME"] = index_name

    click.echo(f"Starting Canopy server on {host}:{port}")
    start_server(host, port=port, reload=reload, config_file=config)


@cli.command(
    help=(
        """
        \b
        Stop the Canopy server.

        This command sends a shutdown request to the Canopy server.
        """
    )
)
@click.option("url", "--url", default=DEFAULT_SERVER_URL,
              help=("URL of the Canopy server to use. "
                    f"Defaults to {DEFAULT_SERVER_URL}"))
def stop(url):
    if os.name != "nt":
        # Check if the server was started using Gunicorn
        res = subprocess.run(["pgrep", "-f", "gunicorn canopy_server.app:app"],
                             capture_output=True)
        output = res.stdout.decode("utf-8").split()

        # If Gunicorn was used, kill all Gunicorn processes
        if output:
            msg = ("It seems that Canopy server was launched using Gunicorn.\n"
                   "Do you want to kill all Gunicorn processes?")
            click.confirm(click.style(msg, fg="red"), abort=True)
            try:
                subprocess.run(["pkill", "-f", "gunicorn canopy_server.app:app"],
                               check=True)
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
        Could not find Canopy server on {url}.
        """
        raise CLIError(msg)


@cli.command(
    help=(
        """
        \b
        Open the Canopy server docs.
        """
    )
)
@click.option("--url", default="http://localhost:8000",
              help="Canopy's server url. Defaults to http://localhost:8000")
def api_docs(url):
    import webbrowser

    generated_docs = False
    try:
        check_server_health(url)
    except CLIError:
        msg = ("Canopy server is not running. Would you like to generate the docs "
               "to a local HTML file?")
        click.confirm(click.style(msg, fg="red"), abort=True)
        generated_docs = True

    if generated_docs:
        import json
        from canopy_server._redocs_template import HTML_TEMPLATE
        from canopy_server.app import app, _init_routes
        # generate docs
        _init_routes(app)
        filename = "canopy-api-docs.html"
        msg = f"Generating docs to {filename}"
        click.echo(click.style(msg, fg="green"))
        with open(filename, "w") as fd:
            print(HTML_TEMPLATE % json.dumps(app.openapi()), file=fd)
        webbrowser.open('file://' + os.path.realpath(filename))
    else:
        webbrowser.open(urljoin(url, "redoc"))


if __name__ == "__main__":
    cli()
