"""
Topic: Document Loaders — WebBaseLoader

Loads a live webpage's text content as a Document -- useful for
pulling in reference material (docs, competitor pages, a client's own
website) without manually copy-pasting it.
"""

from langchain_community.document_loaders import WebBaseLoader

URL = "https://python.langchain.com/docs/concepts/document_loaders/"

loader = WebBaseLoader(URL)
docs = loader.load()

print(f"Loaded {len(docs)} document(s) from {URL}")
print("\nContent preview:")
print(docs[0].page_content[:300].strip())

# Try this next: point this at a client's own "About" page and extract
# their tone of voice before writing copy for them.
