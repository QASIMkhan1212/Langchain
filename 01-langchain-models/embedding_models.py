"""
Topic: Models — Embedding Models

An embedding model turns text into a vector (list of numbers) that
captures meaning -- similar text produces similar vectors. This is the
foundation RAG and semantic search are built on (covered properly in
07-langchain-document-loaders and 08-langchain-text-splitters).

Different example than the usual "compare two sentences" demo: here
we match an incoming client message against 3 of your own service
offerings, to see which one it's actually about.
"""

import numpy as np
from shared_llm import get_embeddings

SERVICES = [
    "WhatsApp AI customer support automation using n8n and the WhatsApp Business API",
    "Instagram and LinkedIn lead generation workflows that scrape and qualify leads automatically",
    "Custom desktop inventory management systems built with Python and SQLite",
]

INCOMING_MESSAGE = "Hi, I run a small pharmacy and need something to track my stock and medicine expiry dates."


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


embeddings = get_embeddings()

service_vectors = embeddings.embed_documents(SERVICES)
message_vector = embeddings.embed_query(INCOMING_MESSAGE)

print(f"Incoming message: {INCOMING_MESSAGE}\n")
scores = [(SERVICES[i], cosine_similarity(message_vector, v)) for i, v in enumerate(service_vectors)]
scores.sort(key=lambda x: x[1], reverse=True)

for service, score in scores:
    print(f"{score:.4f}  {service}")

print(f"\nBest match: {scores[0][0]}")

# Try this next: add a 4th, deliberately unrelated service ("cartoon
# YouTube channel automation") and confirm it scores lowest -- that's
# the sanity check that embeddings are actually capturing meaning.
