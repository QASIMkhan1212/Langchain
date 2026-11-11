"""
Step 2 of the capstone — the agent itself.

This single file exercises almost every topic in the roadmap:
  - prompts        : system prompt
  - tool-calling    : retriever tool + save_note tool, create_agent()
  - runnables       : the retriever itself is a Runnable under the hood
  - structured-output: final ClientBrief extraction via with_structured_output
Document loading / splitting / embedding already happened in ingest.py
-- run that first.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent

from shared_llm import get_model, get_embeddings

INDEX_DIR = "faiss_index"

# --- Load the vector store built by ingest.py ---
embeddings = get_embeddings()
vectorstore = FAISS.load_local(
    INDEX_DIR, embeddings, allow_dangerous_deserialization=True
)

client_notes: list[str] = []  # in-memory notes the agent saves during the chat


# --- Tools ---
@tool
def search_documents(query: str) -> str:
    """Search the client's documents for information relevant to the query.
    Use this whenever the user asks something that might be answered by
    their uploaded PDF/web content."""
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "No relevant documents found."
    return "\n---\n".join(r.page_content for r in results)


@tool
def save_client_note(note: str) -> str:
    """Save an important requirement, preference, or fact the client
    mentioned, so it isn't lost by the end of the conversation."""
    client_notes.append(note)
    return f"Saved note: {note}"


tools = [search_documents, save_client_note]


# --- Agent ---
model = get_model(temperature=0.3)
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=(
        "You are a client research assistant. Answer questions using "
        "search_documents when relevant. When the client states a "
        "requirement, preference, or constraint, call save_client_note "
        "to record it. Be concise."
    ),
)


# --- Structured output: summarize the whole conversation at the end ---
class ClientBrief(BaseModel):
    project_type: Optional[str] = Field(default=None, description="What kind of project this is, if determinable")
    key_requirements: list[str] = Field(description="Concrete requirements mentioned")
    open_questions: list[str] = Field(description="Things still unclear and worth following up on")
    summary: str = Field(description="2-3 sentence summary of the whole conversation")


def generate_brief(transcript: str) -> ClientBrief:
    brief_model = get_model(temperature=0).with_structured_output(ClientBrief)
    return brief_model.invoke(
        f"Based on this conversation and these saved notes, produce a client brief.\n\n"
        f"Conversation:\n{transcript}\n\nSaved notes:\n{chr(10).join(client_notes)}"
    )


def main():
    print("Client Research Assistant. Ask about the ingested docs, or just "
          "chat about a project. Type 'done' to end and get a structured brief.\n")

    messages = []
    transcript_lines = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "done":
            break

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]  # carries tool calls + results forward too

        final_answer = messages[-1].content
        print(f"Assistant: {final_answer}\n")

        transcript_lines.append(f"User: {user_input}")
        transcript_lines.append(f"Assistant: {final_answer}")

    print("\n--- Generating client brief ---\n")
    brief = generate_brief("\n".join(transcript_lines))
    print(brief.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

# Try this next: persist `messages` to disk between runs (reuse the
# JSON pattern from 02-langchain-prompts/project.py), and
# add a 3rd tool that writes the final ClientBrief straight to a file.
