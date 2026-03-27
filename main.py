"""
main.py — FastAPI server for the portfolio agent
=================================================
Endpoints
---------
  POST /chat/stream   — SSE stream; fires one event per graph node
  POST /chat          — simple JSON (non-streaming) fallback
  DELETE /chat/{sid}  — clear Redis session
  GET  /projects      — serve projects.json for the frontend
  GET  /health        — uptime check
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from agent import run_agent, clear_session

load_dotenv()

import os

PROJECTS_JSON_PATH = os.getenv("PROJECTS_JSON_PATH", "")

app = FastAPI(title="Portfolio Agent API", version="1.0.0")

# ── CORS must be registered FIRST, before any other middleware ──
# FastAPI middleware runs in reverse registration order.
# If SlowAPI is added before CORS, it intercepts OPTIONS preflight
# requests first and can return 400 before CORS headers are added.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten to your domain in production
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter — registered AFTER CORS so preflight passes cleanly
limiter = Limiter(key_func=get_remote_address, default_limits=[])
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda r, e: JSONResponse({"error": "rate limit exceeded"}, status_code=429)
)
app.add_middleware(SlowAPIMiddleware)


# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str | None = None   # frontend passes this to maintain memory
    question: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    trace: list[str]


# ─────────────────────────────────────────────
# SSE helpers
# ─────────────────────────────────────────────
def sse_event(event: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_agent(session_id: str, question: str) -> AsyncGenerator[str, None]:
    """
    Run each graph node in a thread and yield SSE events as each
    node completes, so the frontend trace bar lights up in real time.
    """
    yield sse_event("node_start", {"node": "input_node"})
    await asyncio.sleep(0.05)

    yield sse_event("node_start", {"node": "retrieve_node"})
    await asyncio.sleep(0.05)

    loop = asyncio.get_event_loop()

    import time
    t0 = time.time()
    result = await loop.run_in_executor(None, run_agent, session_id, question)
    elapsed = round(time.time() - t0, 2)

    yield sse_event("node_start", {"node": "generate_node"})
    await asyncio.sleep(0.08)

    yield sse_event("node_start", {"node": "output_node"})
    await asyncio.sleep(0.05)

    yield sse_event("answer", {
        "session_id": session_id,
        "answer":     result["answer"],
        "trace":      result["trace"],
        "elapsed_s":  elapsed,
    })

    yield sse_event("done", {"status": "complete"})


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "agent": "portfolio-langgraph-v1"}


@app.get("/projects")
async def get_projects():
    """Serve the full projects.json — used by the frontend to render project cards."""
    path = Path(__file__).parent / PROJECTS_JSON_PATH
    return JSONResponse(content=json.loads(path.read_text()))


@app.post("/chat/stream")
@limiter.limit("10/minute;100/day")
async def chat_stream(request: Request, req: ChatRequest):
    """
    SSE endpoint. Each LangGraph node fires an event so the
    frontend trace bar animates in sync with real execution.

    Events emitted (in order):
      node_start  { node: "input_node" | "retrieve_node" | "generate_node" | "output_node" }
      answer      { session_id, answer, trace, elapsed_s }
      done        { status: "complete" }
    """
    session_id = req.session_id or str(uuid.uuid4())

    return StreamingResponse(
        stream_agent(session_id, req.question),
        media_type="text/event-stream",
        headers={
            # Do NOT manually set Access-Control-Allow-Origin here.
            # CORSMiddleware already adds it. Duplicating the header
            # causes browsers to see two values and reject the response.
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection":        "keep-alive",
        },
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute;100/day")
async def chat(request: Request, req: ChatRequest):
    """Non-streaming fallback for environments that don't support SSE."""
    session_id = req.session_id or str(uuid.uuid4())
    loop       = asyncio.get_event_loop()
    result     = await loop.run_in_executor(None, run_agent, session_id, req.question)
    return ChatResponse(
        session_id=session_id,
        answer=result["answer"],
        trace=result["trace"],
    )


@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    """Clear Redis conversation memory for this session."""
    clear_session(session_id)
    return {"cleared": session_id}
