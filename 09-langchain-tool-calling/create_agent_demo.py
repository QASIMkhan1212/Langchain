"""
Step 4 — Same task, using create_agent().

create_agent() (current LangChain, from the `langchain` package) builds
and runs the exact loop you wrote by hand in manual_tool_loop.py — but
the loop itself is a compiled LangGraph graph under the hood, with
hooks (middleware) for things like retries, fallbacks, and guardrails
that would be painful to bolt onto a raw while-loop yourself.

This is the seam: everything above this file is "LangChain", this file
is where LangGraph quietly takes over the execution.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent

from tools import all_tools

load_dotenv()

agent = create_agent(
    model="groq:llama-3.3-70b-versatile",
    tools=all_tools,
    system_prompt="You are a helpful assistant. Use tools when needed.",
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Karachi, and how many words are in that description?",
            }
        ]
    }
)

# result["messages"] is the FULL transcript: your question, the
# AIMessage(s) with tool_calls, the ToolMessage(s) with results, and
# the final AIMessage answer -- same shape you built by hand in Step 3.
for msg in result["messages"]:
    msg.pretty_print()

print("\nFinal answer only:", result["messages"][-1].content)

# Try this next: print(agent.get_graph().draw_ascii()) to literally see
# the graph create_agent() built for you -- that's your first LangGraph
# visualization, before you've written a single line of LangGraph code.
