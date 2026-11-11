"""
Topic: Document Loaders — TextLoader

Loads a single plain-text file into a list of Document objects (just
1 Document for a single file). Every loader in this folder produces
the same Document shape: .page_content (the text) + .metadata (source
info) -- that consistency is the whole point of using loaders instead
of hand-rolling file-reading code per format.
"""

from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample_data/service_catalog.txt")
docs = loader.load()

print(f"Loaded {len(docs)} document(s)")
print("Metadata:", docs[0].metadata)
print("\nContent preview:")
print(docs[0].page_content[:200])
