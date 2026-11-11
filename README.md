# LangChain Core → LangGraph Learning Path

A self-built, numbered curriculum: one folder per topic, each with
concept scripts PLUS a hands-on project, all running on free-tier
providers (Groq / Gemini) — no paid API key needed anywhere.

## Folder order
```
01-langchain-models            Chat models, embeddings, provider-swapping
02-langchain-prompts           PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
03-langchain-structured-output TypedDict / Pydantic / with_structured_output()
04-langchain-output-parsers    Str / JSON / Pydantic / (legacy) Structured parsers
05-langchain-runnables         LCEL: Sequence, Parallel, Branch, Lambda
06-langchain-chains            Simple / sequential / parallel / conditional chains
07-langchain-document-loaders  Text / CSV / PDF / Directory / Web loaders
08-langchain-text-splitters    Length / structure / markdown / code / semantic splitting
09-langchain-tool-calling      @tool, bind_tools(), manual agent loop, create_agent()
10-langchain-capstone          Everything above combined: a RAG + tool-calling agent
```

Go through 01 → 10 in order. Each of 01-08 is self-contained: its own
`requirements.txt`, `.env.example`, and `shared_llm.py` (a one-function
`get_model()` / `get_embeddings()` loader so switching providers is a
one-line `.env` change, not a code change).

## Setup (per folder, or once if you symlink/reuse the venv)
```
cd 01-langchain-models      # or whichever folder
pip install -r requirements.txt
cp .env.example .env        # fill in GROQ_API_KEY and/or GOOGLE_API_KEY
```
Both are free: Groq at console.groq.com/keys, Gemini at
aistudio.google.com/apikey.

## Inside each of 01-08
- A few short scripts, one per concept, each independently runnable
- `project.py` — the hands-on exercise for that topic (different
  domain/prompts each time, not repeated examples)

## What's in 09 and 10
- `09-langchain-tool-calling` — the one topic not covered by a plain
  "concept file" structure: 4 ordered scripts walk from defining tools
  to hand-building the agent loop to using `create_agent()`
- `10-langchain-capstone` — a Client Research Assistant: ingests docs
  into a vector store, then a tool-calling agent answers questions via
  retrieval and extracts a structured brief at the end. Combines every
  topic from 01-09.

## After this: LangGraph
`create_agent()` (used in 09 and 10) runs on the LangGraph runtime
under the hood. The natural next step: rebuild the capstone's agent as
an explicit `StateGraph`, with the same tools and system prompt, but
with the node/edge wiring written by hand.

## One thing worth knowing
LangChain hit v1.0 in October 2025 — a breaking release. `AgentExecutor`,
`create_react_agent`, and the old `langchain.schema.*` import paths
moved to `langchain-classic`. Everything in this repo is already
written against the current API (`langchain_core.runnables`,
`langchain_text_splitters`, `create_agent` from `langchain.agents`), so
if you see older tutorials using those legacy paths, that's why they
look different from what's here.
