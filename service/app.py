import os
import logging
import sys
import uuid

from context_engine.llm import BaseLLM
from context_engine.llm.models import UserMessage
from context_engine.llm.openai import OpenAILLM
from context_engine.knoweldge_base.tokenizer import OpenAITokenizer, Tokenizer
from context_engine.knoweldge_base import KnowledgeBase
from context_engine.context_engine import ContextEngine
from context_engine.chat_engine import ChatEngine
from starlette.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from fastapi import FastAPI, HTTPException, Body
import uvicorn
from typing import cast
from dotenv import load_dotenv

from context_engine.models.api_models import StreamingChatResponse, ChatResponse
from context_engine.models.data_models import Context
from service.api_models import \
    ChatRequest, ContextQueryRequest, ContextUpsertRequest, HealthStatus

load_dotenv()
INDEX_NAME = os.getenv("INDEX_NAME")
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


@app.get(
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
            namespace=request.namespace,
            batch_size=request.batch_size)

        return upsert_results

    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Internal Service Error: {str(e)}")


@app.get(
    "/health",
)
async def health_check():
    try:
        await run_in_threadpool(kb.connect, force=True)
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


@app.on_event("startup")
async def startup():
    _init_logging()
    _init_engines()


def _init_logging():
    global logger

    file_handler = logging.FileHandler(
        filename=os.getenv("CE_LOG_FILENAME", "context_engine.log")
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
    global kb, context_engine, chat_engine, llm
    Tokenizer.initialize(OpenAITokenizer, model_name='gpt-3.5-turbo-0613')

    if not INDEX_NAME:
        raise ValueError("INDEX_NAME environment variable must be set")

    kb = KnowledgeBase(index_name_suffix=INDEX_NAME)
    kb.connect()
    context_engine = ContextEngine(knowledge_base=kb)
    llm = OpenAILLM(model_name='gpt-3.5-turbo-0613')

    chat_engine = ChatEngine(llm=llm, context_engine=context_engine)


def start(host="0.0.0.0", port=8000, reload=False):
    uvicorn.run("service.app:app",
                host=host, port=port, reload=reload)


if __name__ == "__main__":
    start()
