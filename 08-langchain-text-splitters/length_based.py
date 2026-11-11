"""
Topic: Text Splitters — Length-based (CharacterTextSplitter)

Splits purely by character count, with no awareness of sentence or
paragraph boundaries -- the crudest splitter, but the fastest and
simplest. Useful as a baseline before reaching for something smarter.
"""

from langchain_text_splitters import CharacterTextSplitter

TEXT = """Automation Studio builds custom workflows for small businesses.
Common requests include WhatsApp bots, lead scraping, and inventory
systems. Most projects take 1-3 weeks and are built with n8n, Python,
and free-tier LLM APIs to keep costs low for clients who are often
running a business on a tight budget themselves."""

splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=80,
    chunk_overlap=10,
)
chunks = splitter.split_text(TEXT)

print(f"{len(chunks)} chunks:\n")
for i, c in enumerate(chunks):
    print(f"[{i}] ({len(c)} chars) {c!r}")

# Notice some chunks may cut mid-sentence -- that's the tradeoff of
# length-based splitting. text_structure_based.py fixes that.
