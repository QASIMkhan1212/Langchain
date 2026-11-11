"""
Project 1 (models) — Provider comparison.

Run the SAME prompt through Groq and Gemini and compare: output,
latency, and tone. This is the point of langchain-models: the model
object is swappable, everything downstream (prompts, chains, parsers)
stays identical.

Needs: GROQ_API_KEY and/or GOOGLE_API_KEY in .env (either one works,
the script just skips a provider if its key is missing).
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

PROMPT = "Explain what a vector database is, in 3 sentences, for a beginner."


def try_provider(name: str, build_fn):
    try:
        model = build_fn()
    except Exception as e:
        print(f"[{name}] skipped — {e}")
        return

    start = time.time()
    response = model.invoke(PROMPT)
    elapsed = time.time() - start

    print(f"\n=== {name} ({elapsed:.2f}s) ===")
    print(response.content)


if os.getenv("GROQ_API_KEY"):
    try_provider(
        "Groq (llama-3.3-70b-versatile)",
        lambda: __import__("langchain_groq").ChatGroq(model="llama-3.3-70b-versatile"),
    )
else:
    print("[Groq] skipped — GROQ_API_KEY not set")

if os.getenv("GOOGLE_API_KEY"):
    try_provider(
        "Gemini (gemini-2.0-flash)",
        lambda: __import__("langchain_google_genai").ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
    )
else:
    print("[Gemini] skipped — GOOGLE_API_KEY not set")

# Try this next: swap PROMPT for something more subjective (e.g. "write
# a short poem about debugging") and compare tone, not just correctness.
