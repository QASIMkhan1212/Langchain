# langchain-capstone — Client Research Assistant

Covers every topic from the roadmap in one working pipeline:

| Step | File | Topics used |
|---|---|---|
| 1 | `ingest.py` | document-loaders, text-splitters, embeddings → FAISS |
| 2 | `agent.py` | prompts, tool-calling (`@tool`, `create_agent`), structured-output |

Everything else (models, output-parsers, runnables, chains) is what
these two files are built *on top of* — `get_model()`, the retriever
tool, and `with_structured_output` all use those mechanics internally.

## Setup
```
pip install -r requirements.txt
cp .env.example .env
```
Fill in `GROQ_API_KEY` and `GOOGLE_API_KEY` (both free — Gemini is used
for embeddings since Groq doesn't offer an embeddings endpoint; set
`EMBEDDING_PROVIDER=huggingface` in `.env` instead if you'd rather run
embeddings fully offline, no Google key needed).

## Run
```
python ingest.py   # once, or whenever you add docs to sample_docs/
python agent.py     # chat loop
```

Ask it things like "what document formats does LangChain support?" —
it'll call `search_documents` and answer from the ingested sample PDF
+ webpage. Mention a requirement ("I need this to support French
clients too") and it'll call `save_client_note`. Type `done` to end
and get a structured `ClientBrief` JSON out the other side.

## What's deliberately left simple (so you can extend it)
- Conversation memory is session-only (not saved to disk like
  `02-langchain-prompts/project.py` — add that yourself
  as practice).
- Only one PDF + one webpage are ingested — swap in your own client
  docs in `sample_docs/` and re-run `ingest.py`.

## This is also your LangGraph starting point
`agent.py`'s `create_agent(...)` call is running a LangGraph graph
under the hood. The natural next project: rebuild `agent.py` as an
explicit `StateGraph` — same two tools, same system prompt, same
structured-output step at the end — but with the node/edge wiring
written by you instead of generated for you. That's lesson 1 of
LangGraph, using code you already understand end to end.
