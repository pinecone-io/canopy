from context_engine.knoweldge_base.base_knoweldge_base import BaseKnowledgeBase
from context_engine.llm.openai import OpenAILLM
from context_engine.knoweldge_base.tokenizer import OpenAITokenizer
from context_engine.knoweldge_base.record_encoder import DenseRecordEncoder
from context_engine.knoweldge_base.chunker.token_chunker import TokenChunker
from context_engine.knoweldge_base import KnowledgeBase
from context_engine.context_engine.context_builder import StuffingContextBuilder
from context_engine.context_engine import ContextEngine
from context_engine.chat_engine.query_generator.function_calling \
    import FunctionCallingQueryGenerator
from context_engine.chat_engine.prompt_builder.base import PromptBuilder
from context_engine.chat_engine.history_builder.recent import RecentHistoryBuilder
from context_engine.chat_engine.base import ChatEngine
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from pinecone_text.dense.openai_encoder import OpenAIEncoder
from fastapi import FastAPI, HTTPException, Body
import uvicorn
from typing import Tuple, Iterable
from dotenv import load_dotenv

from context_engine.models.api_models import StreamingChatResponse
from context_engine.service.models import \
    ChatRequest, ContextQueryRequest, ContextUpsertRequest

load_dotenv()


app = FastAPI()


context_engine: ContextEngine
chat_engine: ChatEngine
kb: BaseKnowledgeBase


@app.post(
    "/context/chat/completions",
)
async def chat(
    request: ChatRequest = Body(...),
):
    try:

        answer = await run_in_threadpool(chat_engine.chat,
                                         messages=request.messages,
                                         stream=request.stream)

        if request.stream:

            def generate_streaming_content(responses: Iterable[StreamingChatResponse]):
                for response in responses:
                    data = response.json() + "\n"
                    yield data.encode("utf-8")

            content_stream = generate_streaming_content(answer)  # type: ignore
            return StreamingResponse(content_stream, media_type='text/event-stream')

        else:
            return answer

    except Exception as e:
        print("Error:", e)
        raise HTTPException(
            status_code=500, detail=f"Internal Service Error: {str(e)}")


@app.get(
    "/context/query",
)
async def query(
    request: ContextQueryRequest = Body(...),
):
    try:

        query_results = await run_in_threadpool(
            context_engine.query,
            queries=request.queries,
            max_context_tokens=request.max_tokens)  # type: ignore

        return query_results

    except Exception as e:
        print("Error:", e)
        raise HTTPException(
            status_code=500, detail=f"Internal Service Error: {str(e)}")


@app.post(
    "/context/upsert",
)
async def upsert(
    request: ContextUpsertRequest = Body(...),
):
    try:

        upsert_results = await run_in_threadpool(
            kb.upsert,
            documents=request.documents,
            namespace=request.namespace,
            batch_size=request.batch_size)  # type: ignore

        return upsert_results

    except Exception as e:
        print("Error:", e)
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
    kb, context_engine, chat_engine = _init_engines()


def _init_engines() -> Tuple[BaseKnowledgeBase, ContextEngine, ChatEngine]:
    tokenizer = OpenAITokenizer("gpt-3.5-turbo")
    llm = OpenAILLM("gpt-3.5-turbo",
                    default_max_generated_tokens=256)
    context_builder = StuffingContextBuilder(tokenizer=OpenAITokenizer("gpt-3.5-turbo"),
                                             reference_metadata_field="url")
    history_builder = RecentHistoryBuilder(tokenizer)
    prompt_builder = PromptBuilder(
        context_builder, history_builder, tokenizer, context_ratio=0.7)
    query_generator = FunctionCallingQueryGenerator(
        llm=llm, prompt_builder=prompt_builder, top_k=1000)

    index_name_suffix = "chat-openai-e2e-system-test"

    chunker = TokenChunker(tokenizer=tokenizer,
                           max_chunk_size=256,
                           overlap=30)

    encoder = DenseRecordEncoder(OpenAIEncoder(),
                                 batch_size=500)

    kb = KnowledgeBase(index_name_suffix=index_name_suffix,
                       encoder=encoder,
                       tokenizer=tokenizer,
                       chunker=chunker)
    kb.connect()

    context_engine = ContextEngine(knowledge_base=kb,
                                   context_builder=context_builder)

    system_prompt = """you are a smart assistant that answer questions
 given the context provided by the system. Always provide the reference of
the documents you have used in the format source: <url>"""

    chat_engine = ChatEngine(system_message=system_prompt,
                             llm=llm,
                             knowledge_base=kb,
                             query_builder=query_generator,
                             prompt_builder=prompt_builder,
                             max_prompt_tokens=4096 - 256)

    return kb, context_engine, chat_engine


def start():
    uvicorn.run("context_engine.service.app:app",
                host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
