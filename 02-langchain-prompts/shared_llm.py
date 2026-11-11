"""
Shared model + embeddings loader for the capstone. Same get_model() as
the other repos, plus get_embeddings() -- needed here because this is
the first project that touches a vector store.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_model(temperature: float = 0.7):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)

    if provider == "litellm":
        from langchain_community.chat_models import ChatLiteLLM
        model_name = os.getenv("LITELLM_MODEL", "groq/llama-3.3-70b-versatile")
        return ChatLiteLLM(model=model_name, temperature=temperature)

    raise ValueError(f"Unknown LLM_PROVIDER='{provider}'. Use 'groq', 'gemini', or 'litellm'.")


def get_embeddings():
    """
    Groq doesn't offer an embeddings endpoint, so embeddings are picked
    separately from the chat model:
      - EMBEDDING_PROVIDER=gemini (default): free-tier API, no local download
      - EMBEDDING_PROVIDER=huggingface: fully free/offline, but downloads
        a small (~80MB) model the first time it runs
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    raise ValueError(f"Unknown EMBEDDING_PROVIDER='{provider}'. Use 'gemini' or 'huggingface'.")
