"""
Topic: Text Splitters — Markdown-aware splitting

RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN) knows
Markdown's own structure (headers, code fences, lists) and prefers to
split along those boundaries instead of mid-list or mid-heading.

Example: a client-facing automation proposal written in Markdown.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

MARKDOWN_DOC = """# Project Proposal: Lead Scraper

## Scope
- Scrape listings from 3 real estate sites daily
- Deduplicate against previously seen listings
- Email new matches to the client every morning

## Timeline
Estimated at 10-12 days, including a 2-day testing buffer.

## Pricing
- Setup: PKR 45,000
- Monthly maintenance: PKR 8,000 (optional)
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=150,
    chunk_overlap=0,
)
chunks = splitter.split_text(MARKDOWN_DOC)

print(f"{len(chunks)} chunks:\n")
for i, c in enumerate(chunks):
    print(f"[{i}]\n{c}\n---")
