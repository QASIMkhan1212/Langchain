"""
Topic: Text Splitters — Structure-based (RecursiveCharacterTextSplitter)

The default, go-to splitter. Tries to split on paragraph breaks first,
then sentences, then words -- only falling back to a hard character
cut as a last resort. This is what text-splitters chunking
project (now 08_chunk_and_prep style logic, see project.py in this
folder) actually uses for RAG prep.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXT = """Automation Studio — Client FAQ

Q: How long does a typical project take?
A: Most automation projects take 1-3 weeks depending on complexity.
A simple WhatsApp FAQ bot can be done in under a week, while a
multi-platform lead-generation system usually takes closer to 3 weeks.

Q: What platforms do you use?
A: Mostly n8n for workflow orchestration, Python for custom logic, and
free-tier LLM APIs (Groq, Gemini) to keep ongoing costs low.

Q: Do you offer maintenance after delivery?
A: Yes, optional monthly maintenance is available for most projects."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
)
chunks = splitter.split_text(TEXT)

print(f"{len(chunks)} chunks:\n")
for i, c in enumerate(chunks):
    print(f"[{i}] ({len(c)} chars)\n{c}\n---")

# Compare this output to length_based.py's -- notice chunks here tend
# to break at Q&A boundaries instead of mid-sentence.
