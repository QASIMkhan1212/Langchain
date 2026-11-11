"""
Step 3 — Build the agent loop BY HAND.

This is the step that actually teaches you LangGraph. create_agent()
and LangGraph's prebuilt ReAct node are both just a tidier version of
the exact loop below:

    1. Send messages to the model.
    2. Did it ask for tool call(s)? If no -> done, return its answer.
    3. If yes -> run each requested tool, wrap each result in a
       ToolMessage, append all of them to the message list.
    4. Go back to step 1 with the updated message list.

Once you've written this loop yourself, a LangGraph StateGraph is just
"this same loop, but I get to draw the edges explicitly instead of
them being hardcoded in a while-loop."
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from shared_llm import get_model

from tools import all_tools

load_dotenv()

model = get_model()
model_with_tools = model.bind_tools(all_tools)

# Quick lookup so we can execute whichever tool the model asked for.
tools_by_name = {t.name: t for t in all_tools}


def run_agent(user_input: str, max_steps: int = 5) -> str:
    messages = [
        SystemMessage(content="You are a helpful assistant. Use tools when needed."),
        HumanMessage(content=user_input),
    ]

    for step in range(max_steps):
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # Model answered directly -> loop ends.
            return response.content

        print(f"[step {step}] model requested {len(response.tool_calls)} tool call(s)")

        # Execute every requested tool call and feed results back in.
        for call in response.tool_calls:
            tool = tools_by_name[call["name"]]
            result = tool.invoke(call["args"])
            print(f"    -> {call['name']}({call['args']}) = {result}")

            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )
        # Loop back: the model now sees the tool results and decides
        # whether it has enough to answer, or needs another tool call.

    return "Stopped: hit max_steps without a final answer."


if __name__ == "__main__":
    answer = run_agent(
        "What's the weather in Karachi, and how many words are in that description?"
    )
    print("\nFinal answer:", answer)
