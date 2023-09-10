import logging
import os
import uuid

from context_engine.llm.openai import OpenAILLM
from context_engine.knoweldge_base.tokenizer import OpenAITokenizer, Tokenizer
from context_engine.knoweldge_base import KnowledgeBase
from context_engine.context_engine import ContextEngine
from context_engine.chat_engine import ChatEngine
from starlette.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from fastapi import FastAPI, HTTPException, Body
import uvicorn
from typing import Tuple, cast
from dotenv import load_dotenv

from context_engine.models.api_models import StreamingChatResponse, ChatResponse
from context_engine.models.data_models import Context
from main.api_models import \
    ChatRequest, ContextQueryRequest, ContextUpsertRequest

load_dotenv()
INDEX_NAME = os.getenv("INDEX_NAME")


app = FastAPI()


context_engine: ContextEngine
chat_engine: ChatEngine
kb: KnowledgeBase


@app.post(
    "/context/chat/completions",
)
async def chat(
    request: ChatRequest = Body(...),
):
    try:
        session_id = request.user or "None"  # noqa: F841
        question_id = str(uuid.uuid4())
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
            return chat_response.json()

    except Exception as e:
        logging.exception(e)
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
        logging.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Internal Service Error: {str(e)}")


@app.post(
    "/context/upsert",
)
async def upsert(
    request: ContextUpsertRequest = Body(...),
):
    try:
        logging.info(f"Upserting {len(request.documents)} documents")
        upsert_results = await run_in_threadpool(
            kb.upsert,
            documents=request.documents,
            namespace=request.namespace,
            batch_size=request.batch_size)

        return upsert_results

    except Exception as e:
        logging.exception(e)
        # print("Error:", e)
        raise HTTPException(
            status_code=500, detail=f"Internal Service Error: {str(e)}")


@app.get(
    "/ping",
)
async def ping():
    return "pong"


@app.on_event("startup")
async def startup():
    global kb, context_engine, chat_engine
    _init_logging()
    kb, context_engine, chat_engine = _init_engines()


def _init_logging():
    logging.basicConfig(
        format='pid %(processName)s: d%(asctime)s %(levelname)s: %(message)s',
        level=os.getenv("CE_LOG_LEVEL", "INFO").upper(),
        filename=os.getenv("CE_LOG_FILE", "context_engine.log"),
    )


def _init_engines() -> Tuple[KnowledgeBase, ContextEngine, ChatEngine]:
    Tokenizer.initialize(OpenAITokenizer, model_name='gpt-3.5-turbo-0613')

    if not INDEX_NAME:
        raise ValueError("INDEX_NAME environment variable must be set")

    kb = KnowledgeBase(index_name_suffix=INDEX_NAME)
    # kb.create_index(dimension=1536)
    kb.connect()
    context_engine = ContextEngine(knowledge_base=kb)
    llm = OpenAILLM(model_name='gpt-3.5-turbo-0613')

    chat_engine = ChatEngine(llm=llm, context_engine=context_engine)

    return kb, context_engine, chat_engine


def start():
    uvicorn.run("main.app:app",
                host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
