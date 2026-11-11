"""
Topic: Document Loaders — PyPDFLoader

Loads a PDF, ONE Document PER PAGE (not one Document for the whole
PDF) -- so docs[0] is page 1, docs[1] is page 2, etc. Matters once you
get to chunking (08-langchain-text-splitters): page boundaries are
already a natural first split.
"""

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample_data/proposal_template.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} page(s)")
for i, d in enumerate(docs):
    print(f"\n--- Page {i + 1} (metadata: {d.metadata}) ---")
    print(d.page_content[:200])
