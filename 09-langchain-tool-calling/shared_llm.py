"""
Shared model loader — no paid OpenAI key needed anywhere in these projects.

Set LLM_PROVIDER in your .env to "groq" (default), "gemini", or "litellm".
All 8 projects import get_model() from here instead of hardcoding a provider,
so you can switch providers in ONE place.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_model(temperature: float = 0.7):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        # Free tier, generous limits. https://console.groq.com
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)

    if provider == "gemini":
        # Free tier via Google AI Studio. https://aistudio.google.com/apikey
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)

    if provider == "litellm":
        # Routes through litellm's unified interface -- point LITELLM_MODEL
        # at any provider litellm supports, e.g. "groq/llama-3.3-70b-versatile"
        # or "gemini/gemini-2.0-flash". Same underlying free-tier keys apply.
        from langchain_community.chat_models import ChatLiteLLM
        model_name = os.getenv("LITELLM_MODEL", "groq/llama-3.3-70b-versatile")
        return ChatLiteLLM(model=model_name, temperature=temperature)

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. Use 'groq', 'gemini', or 'litellm'."
    )


if __name__ == "__main__":
    model = get_model()
    print(model.invoke("Say hello in 5 words.").content)
