"""
Project 2 (prompts) — Persistent-memory chatbot.

Builds on chat_prompt_template.py + message_placeholder.py in this folder: uses ChatPromptTemplate +
MessagesPlaceholder so the full conversation gets re-sent every turn,
AND saves/loads history to a JSON file so memory survives across runs
(not just across turns in one session).

Run it, chat a bit, Ctrl+C to quit, run it again -- it remembers.
"""

import json
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from shared_llm import get_model

HISTORY_FILE = "chat_history.json"

model = get_model()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly, concise assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | model


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        raw = json.load(f)
    history = []
    for m in raw:
        cls = HumanMessage if m["role"] == "human" else AIMessage
        history.append(cls(content=m["content"]))
    return history


def save_history(history: list) -> None:
    raw = [
        {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
        for m in history
    ]
    with open(HISTORY_FILE, "w") as f:
        json.dump(raw, f, indent=2)


def main():
    history = load_history()
    print(f"Loaded {len(history)} past messages. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break

        response = chain.invoke({"history": history, "input": user_input})
        print(f"Bot: {response.content}\n")

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response.content))
        save_history(history)


if __name__ == "__main__":
    main()

# Try this next: cap history to the last N messages before sending (real
# chat apps do this to control token cost) and compare behavior.
