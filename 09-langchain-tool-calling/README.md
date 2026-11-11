# langchain-tool-calling

The 9th repo in the set, built to fill the one real
gap before LangGraph. Run the files in this order:

| Order | File | What it teaches |
|---|---|---|
| 1 | `tools.py` | How `@tool` turns a function into something a model can call |
| 2 | `bind_tools_inspect.py` | `model.bind_tools()` + reading `response.tool_calls` |
| 3 | `manual_tool_loop.py` | The agent loop, written by hand — model → tool_calls? → execute → feed back → repeat |
| 4 | `create_agent_demo.py` | Same task via `create_agent()` — compare its output to Step 3's |

## Setup
```
pip install langchain langchain-core langchain-groq langchain-google-genai python-dotenv
```
Create a `.env` file with `OPENAI_API_KEY=...` (same as the other repos).

## What to actually look at
- In Step 2, print `response.tool_calls` for both a question that needs
  a tool and one that doesn't. Confirm the list is empty in the second case.
- In Step 3, put a breakpoint or print statement right before the loop
  re-invokes the model, and read the full `messages` list. That list
  *is* the state LangGraph will manage for you later.
- In Step 4, compare the printed transcript to what you built manually.
  They should be functionally identical.

## After this: where LangGraph actually starts
Once Steps 3 and 4 feel like the same thing to you, you're ready. The
next project is deliberately NOT create_agent — go build the same
tool-loop as an explicit `StateGraph` with nodes and edges, so you
control the branching yourself instead of relying on the prebuilt loop.
That's lesson 1 of LangGraph.
