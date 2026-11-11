"""
Topic: Models — Chat Models

A Chat Model is the current standard in LangChain (the old plain-text
"LLM" completion style is mostly legacy now — most providers have
moved entirely to a chat/message interface, even for single-turn use).

This script runs the SAME task through two different providers to show
that a Chat Model is a swappable object: everything built on top of it
(prompts, chains, parsers) doesn't care which provider is underneath.

Task used here: draft a short client follow-up message for a freelance
automation gig — deliberately NOT the generic "tell me a joke" style
demo, since that's not what you'll actually use this for.
"""

from shared_llm import get_model

TASK = (
    "Write a 3-sentence follow-up message to a client who hasn't replied "
    "in 5 days about a proposal for an n8n lead-generation workflow. "
    "Friendly, not pushy."
)

print("--- Groq (llama-3.3-70b-versatile) ---")
groq_model = get_model()  # LLM_PROVIDER=groq in .env
print(groq_model.invoke(TASK).content)

print("\n--- Switch provider: just change .env, not the code ---")
print("Set LLM_PROVIDER=gemini in your .env and rerun this file to compare.")

# Try this next: invoke() with a plain string (like above) vs a list of
# messages (see chat_prompt_template.py in 02-langchain-prompts) --
# both work, but a message list is how you add a system role / history.
