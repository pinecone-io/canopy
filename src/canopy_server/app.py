import os
import logging
import signal
import sys
import uuid

import openai
from multiprocessing import current_process, parent_process

import yaml
from dotenv import load_dotenv

from canopy.llm import BaseLLM
from canopy.tokenizer import Tokenizer
from canopy.knowledge_base import KnowledgeBase
from canopy.context_engine import ContextEngine
from canopy.chat_engine import ChatEngine
from starlette.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from fastapi import (
    FastAPI,
    HTTPException,
    Body,
    APIRouter
)
import uvicorn
from typing import cast, Union

from canopy.models.api_models import (
    StreamingChatResponse,
    ChatResponse,
)
from canopy.models.data_models import Context, UserMessage
from .models.v1.api_models import (
    ChatRequest,
    ContextQueryRequest,
    ContextUpsertRequest,
    HealthStatus,
    ContextDeleteRequest,
    ShutdownResponse,
    SuccessUpsertResponse,
    SuccessDeleteResponse,
    ContextResponse,
)

from canopy.llm.openai import OpenAILLM
from canopy_cli.errors import ConfigError
from canopy import __version__


APIChatResponse = Union[ChatResponse, EventSourceResponse]

load_dotenv()  # load env vars before import of openai
openai.api_key = os.getenv("OPENAI_API_KEY")

APP_DESCRIPTION = """
Canopy is an open-source Retrieval Augmented Generation (RAG) framework and context engine built on top of the Pinecone vector database. Canopy enables you to quickly and easily experiment with and build applications using RAG. Start chatting with your documents or text data with a few simple commands.

Canopy provides a configurable built-in server, so you can effortlessly deploy a RAG-powered chat application to your existing chat UI or interface. Or you can build your own custom RAG application using the Canopy library.

## Prerequisites

### Pinecone API key
If you don't have a Pinecone account, you can sign up for a free Starter plan at https://www.pinecone.io/.
To find your Pinecone API key and environment log into Pinecone console (https://app.pinecone.io/). You can access your API key from the "API Keys" section in the sidebar of your dashboard, and find the environment name next to it.

### OpenAI API key
You can find your free trial OpenAI API key https://platform.openai.com/account/api-keys. You might need to log in or register for OpenAI services.
"""  # noqa: E501

API_VERSION = "v1"

# Global variables - Application
app: FastAPI = FastAPI(
    title="Canopy API",
    description=APP_DESCRIPTION,
    version=__version__,
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)
openai_api_router = APIRouter()
context_api_router = APIRouter(prefix="/context")
application_router = APIRouter(tags=["Application"])

# Global variables - Engines
context_engine: ContextEngine
chat_engine: ChatEngine
kb: KnowledgeBase
llm: BaseLLM

# Global variables - Logging
logger: logging.Logger


@openai_api_router.post(
    "/chat/completions",
    response_model=None,
    responses={500: {"description": "Failed to chat with Canopy"}},  # noqa: E501
)
async def chat(
    request: ChatRequest = Body(...),
) -> APIChatResponse:
    """
    Chat with Canopy, using the LLM and context engine, and return a response.

    The request schema follows OpenAI's chat completion API schema: https://platform.openai.com/docs/api-reference/chat/create.
    Note that all fields other than `messages` and `stream` are currently ignored. The Canopy server uses the model parameters defined in the `ChatEngine` config for all underlying LLM calls.

    """  # noqa: E501
    try:
        session_id = request.user or "None"  # noqa: F841
        question_id = str(uuid.uuid4())
        logger.debug(f"Received chat request: {request.messages[-1].content}")
        answer = await run_in_threadpool(
            chat_engine.chat, messages=request.messages, stream=request.stream
        )

        if request.stream:

            def stringify_content(response: StreamingChatResponse):
                for chunk in response.chunks:
                    chunk.id = question_id
                    data = chunk.json()
                    yield data

            content_stream = stringify_content(cast(StreamingChatResponse, answer))
            return EventSourceResponse(content_stream, media_type="text/event-stream")

        else:
            chat_response = cast(ChatResponse, answer)
            chat_response.id = question_id
            return chat_response

    except Exception as e:
        logger.exception(f"Chat with question_id {question_id} failed")
        raise HTTPException(status_code=500, detail=f"Internal Service Error: {str(e)}")


@context_api_router.post(
    "/query",
    response_model=ContextResponse,
    responses={
        500: {"description": "Failed to query the knowledge base or build the context"}
    },
)
async def query(
    request: ContextQueryRequest = Body(...),
) -> ContextResponse:
    """
    Query the knowledge base for relevant context.
    The returned text may be structured or unstructured, depending on the Canopy configuration.
    Query allows limiting the context length in tokens to control LLM costs.
    This method does not pass through the LLM and uses only retrieval and construction from Pinecone DB.
    """  # noqa: E501
    try:
        context: Context = await run_in_threadpool(
            context_engine.query,
            queries=request.queries,
            max_context_tokens=request.max_tokens,
        )
        return ContextResponse(content=context.content.to_text(),
                               num_tokens=context.num_tokens)

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Internal Service Error: {str(e)}")


@context_api_router.post(
    "/upsert",
    response_model=SuccessUpsertResponse,
    responses={500: {"description": "Failed to upsert documents"}},
)
async def upsert(
    request: ContextUpsertRequest = Body(...),
) -> SuccessUpsertResponse:
    """
    Upsert documents into the knowledge base. Upserting is a way to add new documents or update existing ones.
    Each document has a unique ID. If a document with the same ID already exists, it is updated.

    The documents are chunked and encoded, then the resulting encoded chunks are sent to the Pinecone index in batches.
    """  # noqa: E501
    try:
        logger.info(f"Upserting {len(request.documents)} documents")
        await run_in_threadpool(
            kb.upsert, documents=request.documents, batch_size=request.batch_size
        )

        return SuccessUpsertResponse()

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Internal Service Error: {str(e)}")


@context_api_router.post(
    "/delete",
    response_model=SuccessDeleteResponse,
    responses={500: {"description": "Failed to delete documents"}},
)
async def delete(
    request: ContextDeleteRequest = Body(...),
) -> SuccessDeleteResponse:
    """
    Delete documents from the knowledgebase. Deleting documents is done by their unique ID.
    """  # noqa: E501
    try:
        logger.info(f"Delete {len(request.document_ids)} documents")
        await run_in_threadpool(kb.delete, document_ids=request.document_ids)
        return SuccessDeleteResponse()

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Internal Service Error: {str(e)}")


@application_router.get(
    "/health",
    response_model=HealthStatus,
    responses={500: {"description": "Failed to connect to Pinecone or LLM"}},
)
@app.exception_handler(Exception)
async def health_check() -> HealthStatus:
    """
    Health check for the Canopy server. This endpoint checks the connection to Pinecone and the LLM.
    """  # noqa: E501
    try:
        await run_in_threadpool(kb.verify_index_connection)
    except Exception as e:
        err_msg = f"Failed connecting to Pinecone Index {kb._index_name}"
        logger.exception(err_msg)
        raise HTTPException(
            status_code=500, detail=f"{err_msg}. Error: {str(e)}"
        ) from e

    try:
        msg = UserMessage(content="This is a health check. Are you alive? Be concise")
        await run_in_threadpool(llm.chat_completion, messages=[msg], max_tokens=5)
    except Exception as e:
        err_msg = f"Failed to communicate with {llm.__class__.__name__}"
        logger.exception(err_msg)
        raise HTTPException(
            status_code=500, detail=f"{err_msg}. Error: {str(e)}"
        ) from e

    return HealthStatus(pinecone_status="OK", llm_status="OK")


@application_router.get("/shutdown")
async def shutdown() -> ShutdownResponse:
    """
    __WARNING__: Experimental method.


    This method will shutdown the server. It is used for testing purposes, and not recommended to be used
    in production.
    This method will locate the parent process and send a SIGINT signal to it.
    """  # noqa: E501
    logger.info("Shutting down")
    proc = current_process()
    p_process = parent_process()
    pid = p_process.pid if p_process is not None else proc.pid
    if not pid:
        raise HTTPException(
            status_code=500,
            detail="Failed to locate parent process. Cannot shutdown server.",
        )
    if sys.platform == 'win32':
        kill_signal = signal.CTRL_C_EVENT
    else:
        kill_signal = signal.SIGINT
    os.kill(pid, kill_signal)
    return ShutdownResponse()


@app.on_event("startup")
async def startup():
    _init_logging()
    _init_engines()
    _init_routes(app)
    await health_check()


def _init_routes(app):
    # Include the application level router (health, shutdown, ...)
    app.include_router(application_router, include_in_schema=False)
    app.include_router(application_router, prefix=f"/{API_VERSION}")
    # Include the API without version == latest
    app.include_router(context_api_router, include_in_schema=False)
    app.include_router(openai_api_router, include_in_schema=False)
    # Include the API version in the path, API_VERSION should be the latest version.
    app.include_router(context_api_router, prefix=f"/{API_VERSION}", tags=["Context"])
    app.include_router(openai_api_router, prefix=f"/{API_VERSION}", tags=["LLM"])


def _init_logging():
    global logger

    file_handler = logging.FileHandler(
        filename=os.getenv("CE_LOG_FILENAME", "canopy.log")
    )
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        format="%(asctime)s - %(processName)s - %(name)-10s [%(levelname)-8s]:  "
        "%(message)s",
        level=os.getenv("CE_LOG_LEVEL", "INFO").upper(),
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger(__name__)


def _init_engines():
    global kb, context_engine, chat_engine, llm, logger

    index_name = os.getenv("INDEX_NAME")
    if not index_name:
        raise ValueError("INDEX_NAME environment variable must be set")

    config_file = os.getenv("CANOPY_CONFIG_FILE")
    if config_file:
        _load_config(config_file)

    else:
        logger.info(
            "Did not find config file. Initializing engines with default "
            "configuration"
        )
        Tokenizer.initialize()
        kb = KnowledgeBase(index_name=index_name)
        context_engine = ContextEngine(knowledge_base=kb)
        llm = OpenAILLM()
        chat_engine = ChatEngine(context_engine=context_engine, llm=llm)

    kb.connect()


def _load_config(config_file):
    global chat_engine, llm, context_engine, kb, logger
    logger.info(f"Initializing engines with config file {config_file}")
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.exception(f"Failed to load config file {config_file}")
        raise ConfigError(f"Failed to load config file {config_file}. Error: {str(e)}")
    tokenizer_config = config.get("tokenizer", {})
    Tokenizer.initialize_from_config(tokenizer_config)
    if "chat_engine" not in config:
        raise ConfigError(
            f"Config file {config_file} must contain a 'chat_engine' section"
        )
    chat_engine_config = config["chat_engine"]
    try:
        chat_engine = ChatEngine.from_config(chat_engine_config)
    except Exception as e:
        logger.exception(
            f"Failed to initialize chat engine from config file {config_file}"
        )
        raise ConfigError(
            f"Failed to initialize chat engine from config file {config_file}."
            f" Error: {str(e)}"
        )
    llm = chat_engine.llm
    context_engine = chat_engine.context_engine
    kb = context_engine.knowledge_base


def start(host="0.0.0.0", port=8000, reload=False, config_file=None):
    if config_file:
        os.environ["CANOPY_CONFIG_FILE"] = config_file

    uvicorn.run("canopy_server.app:app", host=host, port=port, reload=reload, workers=0)


if __name__ == "__main__":
    start()
