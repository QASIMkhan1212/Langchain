"""
Topic: Text Splitters — Semantic splitting (SemanticChunker)

Instead of a fixed character count, this splitter embeds sentences and
cuts where the MEANING shifts significantly -- so a chunk stays
topically coherent even if that makes it longer or shorter than a
fixed-size chunk would be. Costs more (it calls the embedding model
while splitting), so reserve it for content where topic coherence
actually matters more than speed.
"""

from langchain_experimental.text_splitter import SemanticChunker
from shared_llm import get_embeddings

TEXT = (
    "Automation Studio builds WhatsApp bots and lead-generation workflows "
    "for small businesses. Projects typically take 1 to 3 weeks. "
    "Separately, the studio's founder is also building an ML side project "
    "to predict fuel prices in Pakistan, unrelated to client work. "
    "Client pricing is usually split 50% upfront and 50% on delivery."
)

splitter = SemanticChunker(get_embeddings())
chunks = splitter.split_text(TEXT)

print(f"{len(chunks)} semantic chunk(s):\n")
for i, c in enumerate(chunks):
    print(f"[{i}]\n{c}\n---")

# Notice the unrelated "fuel price ML project" sentence likely ends up
# isolated in its own chunk -- that's the semantic boundary being
# detected, not a fixed character count.
