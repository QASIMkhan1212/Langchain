"""
Topic: Document Loaders — DirectoryLoader

Loads every matching file in a folder at once, using whichever
individual loader you specify as loader_cls -- handy when a client's
notes/docs are spread across several files, not one.

Example: load every call-notes file in sample_data/client_notes/ as
one batch instead of loading each by hand.
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    "sample_data/client_notes",
    glob="*.txt",
    loader_cls=TextLoader,
)
docs = loader.load()

print(f"Loaded {len(docs)} file(s) from the directory\n")
for d in docs:
    print(f"Source: {d.metadata['source']}")
    print(d.page_content)
    print("---")
