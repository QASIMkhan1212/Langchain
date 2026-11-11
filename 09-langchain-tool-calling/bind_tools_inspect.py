"""
Step 2 — Bind tools to the model and inspect what it decides to call.

model.bind_tools([...]) does NOT execute anything. It just attaches the
tool schemas to every request, so the model *can* ask for a tool call
instead of answering directly. The model still only returns TEXT or a
request to call a tool — it never runs the function itself. That's your
job (Step 3).

Run this and look closely at `response.tool_calls` — that's the whole
mental model LangGraph is built on.
"""

from dotenv import load_dotenv
from shared_llm import get_model

from tools import all_tools

load_dotenv()

model = get_model()
model_with_tools = model.bind_tools(all_tools)

# --- Case 1: a question that needs a tool ---
response = model_with_tools.invoke("What's 23 * 7, and also what's the weather in Lahore?")

print("Content (usually empty when tools are called):")
print(repr(response.content))

print("\nTool calls the model asked for:")
for call in response.tool_calls:
    print(f"  - name: {call['name']}")
    print(f"    args: {call['args']}")
    print(f"    id:   {call['id']}")

# --- Case 2: a question that needs NO tool ---
response2 = model_with_tools.invoke("What is LangChain in one sentence?")
print("\n--- No-tool-needed case ---")
print("tool_calls:", response2.tool_calls)  # should be an empty list
print("content:", response2.content)

# Key insight: the model decides per-message whether a tool is needed.
# bind_tools() just gives it the option — it's not forced to use one.
