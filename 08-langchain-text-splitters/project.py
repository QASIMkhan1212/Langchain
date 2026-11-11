"""
Project 8 (text-splitters) — Chunk the documents collected in
07-langchain-document-loaders/project.py.

Loads that sibling folder's project.py directly (since folder names
starting with a digit can't be imported the normal way), runs its
load_all(), then splits the result into model-sized chunks with
RecursiveCharacterTextSplitter -- the exact prep step that comes right
before "embed + store in a vector DB" in a real RAG pipeline (built
out fully in 10-langchain-capstone/).
"""

import importlib.util
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

_SIBLING_PATH = os.path.join(
    os.path.dirname(__file__), "..", "07-langchain-document-loaders", "project.py"
)
_spec = importlib.util.spec_from_file_location("doc_collector", _SIBLING_PATH)
doc_collector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(doc_collector)


def chunk_documents(chunk_size: int = 300, chunk_overlap: int = 50):
    docs = doc_collector.load_all()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    chunks = chunk_documents()
    print(f"\n{len(chunks)} chunks produced:\n")
    for i, c in enumerate(chunks[:5]):  # preview just the first 5
        print(f"[{i}] ({len(c.page_content)} chars, source_type={c.metadata.get('source_type')})")
        print(f"    {c.page_content[:100].strip()}...\n")

    if len(chunks) > 5:
        print(f"...and {len(chunks) - 5} more chunks.")

# Try this next: this is RAG-ready -- see 10-langchain-capstone/ for
# the full embed + store + retrieve pipeline built on exactly this
# chunking step.
