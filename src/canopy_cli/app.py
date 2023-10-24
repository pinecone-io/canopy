import os
import logging
import signal
import sys
import uuid

import openai
from multiprocessing import current_process

import yaml
from dotenv import load_dotenv

from canopy.llm import BaseLLM
from canopy.llm.models import UserMessage
from canopy.tokenizer import Tokenizer
from canopy.knowledge_base import KnowledgeBase
from canopy.context_engine import ContextEngine
from canopy.chat_engine import ChatEngine
from starlette.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from fastapi import FastAPI, HTTPException, Body
import uvicorn
from typing import cast

from canopy.models.api_models import StreamingChatResponse, ChatResponse
from canopy.models.data_models import Context
from canopy_cli.api_models import \
    ChatRequest, ContextQueryRequest, \
    ContextUpsertRequest, HealthStatus, ContextDeleteRequest

from canopy.llm.openai import OpenAILLM
from canopy_cli.errors import ConfigError

load_dotenv()  # load env vars before import of openai
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

context_engine: ContextEngine
chat_engine: ChatEngine
kb: KnowledgeBase
llm: BaseLLM
logger: logging.Logger


@app.post(
    "/context/chat/completions",
)
async def chat(
    request: ChatRequest = Body(...),
):
    try:
        session_id = request.user or "None"  # noqa: F841
        question_id = str(uuid.uuid4())
        logger.debug(f"Received chat request: {request.messages[-1].content}")
        answer = await run_in_threadpool(chat_engine.chat,
                                         messages=request.messages,
                                         stream=request.stream)

        if request.stream:
            def stringify_content(response: StreamingChatResponse):
                for chunk in response.chunks:
                    chunk.id = question_id
                    data = chunk.json()
                    yield data

            content_stream = stringify_content(cast(StreamingChatResponse, answer))
            return EventSourceResponse(content_stream, media_type='text/event-stream')

        else:
            chat_response = cast(ChatResponse, answer)
            chat_response.id = question_id
            return chat_response

    except Exception as e:
        logger.exception(f"Chat with question_id {question_id} failed")
        raise HTTPException(
            status_code=500, detail=f"Internal Service Error: {str(e)}")


@app.post(
    "/context/query",
)
async def query(
    request: ContextQueryRequest = Body(...),
):
    try:
        context: Context = await run_in_threadpool(
            context_engine.query,
            queries=request.queries,
            max_context_tokens=request.max_tokens)

        return context.content

    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Internal Service Error: {str(e)}")


@app.post(
    "/context/upsert",
)
async def upsert(
    request: ContextUpsertRequest = Body(...),
):
    try:
        logger.info(f"Upserting {len(request.documents)} documents")
        upsert_results = await run_in_threadpool(
            kb.upsert,
            documents=request.documents,
            batch_size=request.batch_size)

        return upsert_results

    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Internal Service Error: {str(e)}")


@app.post(
    "/context/delete",
)
async def delete(
    request: ContextDeleteRequest = Body(...),
):
    try:
        logger.info(f"Delete {len(request.document_ids)} documents")
        await run_in_threadpool(
            kb.delete,
            document_ids=request.document_ids)
        return {"message": "success"}

    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Internal Service Error: {str(e)}")


@app.get(
    "/health",
)
async def health_check():
    try:
        await run_in_threadpool(kb.verify_index_connection)
    except Exception as e:
        err_msg = f"Failed connecting to Pinecone Index {kb._index_name}"
        logger.exception(err_msg)
        raise HTTPException(
            status_code=500, detail=f"{err_msg}. Error: {str(e)}") from e

    try:
        msg = UserMessage(content="This is a health check. Are you alive? Be concise")
        await run_in_threadpool(llm.chat_completion,
                                messages=[msg],
                                max_tokens=50)
    except Exception as e:
        err_msg = f"Failed to communicate with {llm.__class__.__name__}"
        logger.exception(err_msg)
        raise HTTPException(
            status_code=500, detail=f"{err_msg}. Error: {str(e)}") from e

    return HealthStatus(pinecone_status="OK", llm_status="OK")


@app.get(
    "/shutdown"
)
async def shutdown():
    logger.info("Shutting down")
    proc = current_process()
    pid = proc._parent_pid if "SpawnProcess" in proc.name else proc.pid
    os.kill(pid, signal.SIGINT)
    return {"message": "Shutting down"}


@app.on_event("startup")
async def startup():
    _init_logging()
    _init_engines()


def _init_logging():
    global logger

    file_handler = logging.FileHandler(
        filename=os.getenv("CE_LOG_FILENAME", "canopy.log")
    )
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        format='%(asctime)s - %(processName)s - %(name)-10s [%(levelname)-8s]:  '
               '%(message)s',
        level=os.getenv("CE_LOG_LEVEL", "INFO").upper(),
        handlers=handlers,
        force=True
    )
    logger = logging.getLogger(__name__)


def _init_engines():
    global kb, context_engine, chat_engine, llm, logger

    index_name = os.getenv("INDEX_NAME")
    if not index_name:
        raise ValueError("INDEX_NAME environment variable must be set")

    config_file = os.getenv("RESIN_CONFIG_FILE")
    if config_file:
        _load_config(config_file)

    else:
        logger.info("Did not find config file. Initializing engines with default "
                    "configuration")
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
        raise ConfigError(
            f"Failed to load config file {config_file}. Error: {str(e)}"
        )
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
        os.environ["RESIN_CONFIG_FILE"] = config_file

    uvicorn.run("canopy_cli.app:app", host=host, port=port, reload=reload, workers=0)


if __name__ == "__main__":
    start()
