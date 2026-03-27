"""
agent.py — LangGraph portfolio agent
=====================================
Graph topology:
    input_node → retrieve_node → generate_node → output_node

Each node emits a Server-Sent Event so the frontend trace bar
lights up in real time as the graph executes.

State schema
------------
  session_id   : str        — Redis key prefix for conversation memory
  question     : str        — raw user input
  context      : str        — retrieved knowledge from projects.json
  answer       : str        — generated response
  trace        : list[str]  — nodes visited so far
  history      : list[dict] — [{role, content}] conversation turns
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TypedDict, Annotated

import redis
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
REDIS_URL          = os.getenv("REDIS_URL", "redis://localhost:6379")
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL          = os.getenv("LLM_MODEL", "llama3-8b-8192")
PROJECTS_JSON_PATH = os.getenv("PROJECTS_JSON_PATH", "../projects.json")
SESSION_TTL        = 60 * 60 * 1  # 1 hour

# ─────────────────────────────────────────────
# Redis client (lazy singleton)
# ─────────────────────────────────────────────
_redis_client: redis.Redis | None = None

def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


# ─────────────────────────────────────────────
# Knowledge base — loaded once at startup
# ─────────────────────────────────────────────
def load_knowledge_base() -> str:
    """
    Read projects.json and flatten it into a rich text
    knowledge base the LLM can reason over.
    """
    path = Path(__file__).parent / PROJECTS_JSON_PATH
    data = json.loads(path.read_text())

    owner   = data["owner"]
    lines: list[str] = []

    # Owner bio
    lines.append(f"=== OWNER ===")
    lines.append(f"Name: {owner['name']}")
    lines.append(f"Title: {owner['title']}")
    lines.append(f"Location: {owner['location']}")
    lines.append(f"Bio: {owner['bio']}")
    lines.append(f"Skills: {', '.join(owner['skills'])}")
    lines.append("")

    # Projects
    lines.append("=== PROJECTS ===")
    for p in data["projects"]:
        lines.append(f"--- {p['name']} ({p['type']}, {p['status']}) ---")
        lines.append(f"Description: {p['description']}")
        lines.append(f"Architecture: {p['architecture']}")
        lines.append(f"Tech stack: {', '.join(p['tech'])}")
        if p.get("github"):
            lines.append(f"GitHub: {p['github']}")
        if p.get("demo"):
            lines.append(f"Demo: {p['demo']}")
        lines.append("")

    # Experience
    lines.append("=== EXPERIENCE ===")
    for e in data["experience"]:
        lines.append(f"{e['role']} @ {e['company']} ({e['period']})")
        lines.append(f"  {e['description']}")
    lines.append("")

    return "\n".join(lines)


KNOWLEDGE_BASE: str = load_knowledge_base()

SYSTEM_PROMPT = f"""You are the portfolio AI agent for {json.loads((Path(__file__).parent / PROJECTS_JSON_PATH).read_text())['owner']['name']}.

You have access to their complete professional knowledge base:

{KNOWLEDGE_BASE}

Instructions:
- Answer questions about their projects, skills, experience, and background.
- Be specific — quote metrics, tech stack choices, and architecture decisions from the knowledge base.
- Keep answers to 2-4 sentences unless a detailed breakdown is requested.
- If asked something outside the portfolio (e.g. general coding help), politely redirect.
- Sound like a knowledgeable assistant, not a corporate chatbot.
"""


# ─────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────
def get_llm():
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.3,
            streaming=True
        )

    if LLM_PROVIDER == "groq":
        return ChatGroq(
            model=LLM_MODEL,
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


# ─────────────────────────────────────────────
# Graph State
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    session_id: str
    question:   str
    context:    str
    answer:     str
    trace:      list[str]
    history:    list[dict]   # [{role: "user"|"assistant", content: str}]


# ─────────────────────────────────────────────
# Redis helpers for conversation memory
# ─────────────────────────────────────────────
def _history_key(session_id: str) -> str:
    return f"portfolio:history:{session_id}"


def load_history(session_id: str) -> list[dict]:
    r = get_redis()
    raw = r.get(_history_key(session_id))
    if raw:
        return json.loads(raw)
    return []


def save_history(session_id: str, history: list[dict]) -> None:
    r = get_redis()
    r.setex(_history_key(session_id), SESSION_TTL, json.dumps(history))


# ─────────────────────────────────────────────
# Node 1 — input_node
# Validates and pre-processes the incoming question.
# Loads conversation history from Redis.
# ─────────────────────────────────────────────
def input_node(state: AgentState) -> AgentState:
    question = state["question"].strip()
    if not question:
        question = "Tell me about yourself."

    # Load prior turns from Redis
    history = load_history(state["session_id"])

    return {
        **state,
        "question": question,
        "history":  history,
        "trace":    ["input_node"],
        "context":  "",
        "answer":   "",
    }


# ─────────────────────────────────────────────
# Node 2 — retrieve_node
# Keyword-search the knowledge base to pull the most
# relevant sections for this question.
# (In a full deployment, replace with vector search.)
# ─────────────────────────────────────────────
SECTIONS = {
    "projects": ["project", "built", "shipped", "work", "github", "demo",
                 "system", "pipeline", "agent", "rag", "llm", "langraph",
                 "langchain", "architecture", "stack", "tech"],
    "experience": ["experience", "job", "career", "company", "role",
                   "worked", "team", "engineer", "years"],
    "skills": ["skill", "know", "technologies", "expertise", "capable",
               "proficient", "familiar", "use"],
    "owner": ["who", "about", "yourself", "background", "bio",
              "contact", "email", "hire", "available"],
}

def retrieve_node(state: AgentState) -> AgentState:
    print("agent is in retrieve_node\n")
    q_lower = state["question"].lower()
    data    = json.loads(
        (Path(__file__).parent / PROJECTS_JSON_PATH).read_text()
    )

    relevant_chunks: list[str] = []

    # Always include owner bio
    owner = data["owner"]
    relevant_chunks.append(
        f"Owner: {owner['name']}, {owner['title']}. {owner['bio']}"
    )

    # Score each project by keyword overlap with question
    scored_projects: list[tuple[int, dict]] = []
    for p in data["projects"]:
        score = 0
        searchable = (
            p["name"] + " " + p["description"] + " " +
            " ".join(p["tech"]) + " " + p["architecture"]
        ).lower()
        for word in q_lower.split():
            if len(word) > 3 and word in searchable:
                score += 1
        # Boost if project name is directly mentioned
        if p["name"].lower() in q_lower:
            score += 5
        scored_projects.append((score, p))

    scored_projects.sort(key=lambda x: x[0], reverse=True)

    # Include top-3 projects always; all if score > 0
    for i, (score, p) in enumerate(scored_projects):
        if i < 3 or score > 0:
            relevant_chunks.append(
                f"Project: {p['name']} ({p['type']}, {p['status']})\n"
                f"  Description: {p['description']}\n"
                f"  Architecture: {p['architecture']}\n"
                f"  Tech: {', '.join(p['tech'])}\n"
            )

    # Include experience if relevant
    for keyword in SECTIONS["experience"]:
        if keyword in q_lower:
            for e in data["experience"]:
                relevant_chunks.append(
                    f"Experience: {e['role']} @ {e['company']} "
                    f"({e['period']}): {e['description']}"
                )
            break

    context = "\n\n".join(relevant_chunks)

    return {
        **state,
        "context": context,
        "trace":   state["trace"] + ["retrieve_node"],
    }


# ─────────────────────────────────────────────
# Node 3 — generate_node
# Calls the LLM with full context + conversation history.
# Saves the updated history back to Redis.
# ─────────────────────────────────────────────
def generate_node(state: AgentState) -> AgentState:
    llm = get_llm()
    print("agent is in generate_node\n")
    # Build message list
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Inject prior conversation turns (last 6 max to keep context tight)
    for turn in state["history"][-6:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    # Add retrieved context + current question
    messages.append(HumanMessage(content=(
        f"Relevant context:\n{state['context']}\n\n"
        f"Question: {state['question']}"
    )))

    response  = llm.invoke(messages)
    answer    = response.content

    # Update conversation history and persist to Redis
    new_history = state["history"] + [
        {"role": "user",      "content": state["question"]},
        {"role": "assistant", "content": answer},
    ]
    save_history(state["session_id"], new_history)

    return {
        **state,
        "answer":  answer,
        "history": new_history,
        "trace":   state["trace"] + ["generate_node"],
    }


# ─────────────────────────────────────────────
# Node 4 — output_node
# Final pass — trims whitespace, marks trace complete.
# ─────────────────────────────────────────────
def output_node(state: AgentState) -> AgentState:
    return {
        **state,
        "answer": state["answer"].strip(),
        "trace":  state["trace"] + ["output_node"],
    }


# ─────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("input_node",    input_node)
    g.add_node("retrieve_node", retrieve_node)
    g.add_node("generate_node", generate_node)
    g.add_node("output_node",   output_node)

    g.set_entry_point("input_node")
    g.add_edge("input_node",    "retrieve_node")
    g.add_edge("retrieve_node", "generate_node")
    g.add_edge("generate_node", "output_node")
    g.add_edge("output_node",   END)

    return g.compile()


# Compiled graph — singleton
GRAPH = build_graph()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def run_agent(session_id: str, question: str) -> dict:
    """
    Run the graph synchronously and return the full result dict.
    Used by the SSE endpoint to stream node-by-node updates.
    """
    result = GRAPH.invoke({
        "session_id": session_id,
        "question":   question,
        "context":    "",
        "answer":     "",
        "trace":      [],
        "history":    [],
    })
    return result


def clear_session(session_id: str) -> None:
    get_redis().delete(_history_key(session_id))
