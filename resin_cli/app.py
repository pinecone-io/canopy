import json
import os
import logging
import uuid

from dotenv import load_dotenv

from oplog import OperationHandler, Operation
from oplog.formatters import BaseOperationFormatter

from resin.llm import BaseLLM
from resin.llm.models import UserMessage
from resin.tokenizer import OpenAITokenizer, Tokenizer
from resin.knoweldge_base import KnowledgeBase
from resin.context_engine import ContextEngine
from resin.chat_engine import ChatEngine
from starlette.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from fastapi import FastAPI, HTTPException, Body
import uvicorn
from typing import cast

from resin.models.api_models import StreamingChatResponse, ChatResponse
from resin.models.data_models import Context
from resin_cli.api_models import \
    ChatRequest, ContextQueryRequest, ContextUpsertRequest, HealthStatus

load_dotenv()  # load env vars before import of openai
from resin.llm.openai import OpenAILLM  # noqa: E402


class JsonFormatter(BaseOperationFormatter):
    def format_op(self, op: Operation) -> str:
        optional_fields = {
            "exception_type": str(op.exception_type or ""),
            "traceback": str(op.traceback or ""),
            "custom_props": op.custom_props,
            "global_props": op.global_props,
        }

        row = {
            "start_time_utc": op.start_time_utc_str,
            "name": op.name,
            "result": op.result,
            "duration_ms": str(op.duration_ms),
            "correlation_id": op.correlation_id,
            **{k: v for k, v in optional_fields.items() if v},
        }

        return json.dumps(row, ensure_ascii=False)


json_op_handler = OperationHandler(
    handler=logging.FileHandler(filename=os.getenv("CE_LOG_FILENAME", "resin.logs")),
    formatter=JsonFormatter(),
)
logging.basicConfig(level=os.getenv("CE_LOG_LEVEL", "INFO").upper(),
                    handlers=[json_op_handler])


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
        with Operation(name="chat") as op:
            session_id = request.user or "None"  # noqa: F841
            question_id = str(uuid.uuid4())

            op.add("content", request.messages[-1].content)
            op.add("question_id", question_id)

            answer = await run_in_threadpool(chat_engine.chat,
                                             messages=request.messages,
                                             stream=request.stream)
            if request.stream:
                def stringify_content(response: StreamingChatResponse):
                    entire_answer = ""
                    for chunk in response.chunks:
                        chunk.id = question_id
                        data = chunk.json()
                        yield data
                        entire_answer += chunk.choices[0].delta.get("content", "")
                    op.add("answer", entire_answer)

                content_stream = stringify_content(cast(StreamingChatResponse, answer))
                return EventSourceResponse(content_stream, media_type='text/event-stream')
            else:
                chat_response = cast(ChatResponse, answer)
                chat_response.id = question_id
                op.add("answer", chat_response.choices[0].message.content)
                return chat_response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal Service Error: {str(e)}")


@app.get(
    "/context/query",
)
async def query(
    request: ContextQueryRequest = Body(...),
):
    try:
        with Operation(name="query") as op:
            op.add("max_context_tokens", request.max_tokens)
            op.add("queries", [q.text for q in request.queries])
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
        with Operation(name="upsert") as op:
            op.add("num_documents", len(request.documents))
            op.add("batch_size", request.batch_size)
            op.add("namespace", request.namespace)

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
        with Operation(name="health_check_kb_connection"):
            await run_in_threadpool(kb.verify_connection_health)
    except Exception as e:
        err_msg = f"Failed connecting to Pinecone Index {kb._index_name}"
        logger.exception(err_msg)
        raise HTTPException(
            status_code=500, detail=f"{err_msg}. Error: {str(e)}") from e

    try:
        with Operation(name="health_check_chat"):
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
    _init_engines()
    pass


def _init_engines():
    global kb, context_engine, chat_engine, llm
    Tokenizer.initialize(OpenAITokenizer, model_name='gpt-3.5-turbo-0613')

    if not INDEX_NAME:
        raise ValueError("INDEX_NAME environment variable must be set")

    kb = KnowledgeBase(index_name=INDEX_NAME)
    context_engine = ContextEngine(knowledge_base=kb)
    llm = OpenAILLM()

    chat_engine = ChatEngine(context_engine=context_engine, llm=llm)


def start(host="0.0.0.0", port=8000, reload=False):
    uvicorn.run("resin_cli.app:app",
                host=host, port=port, reload=reload)


if __name__ == "__main__":
    start()
