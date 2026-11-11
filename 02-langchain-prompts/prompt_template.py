"""
Topic: Prompts — PromptTemplate

A PromptTemplate is a reusable string with {placeholders}, so the same
instructions can be filled in with different values without rewriting
the prompt text each time. Also demonstrates save/load from a JSON
file, so a prompt can be version-controlled or shared between projects.

Example here: generating a quote message for an automation service --
different scenario than the usual "write a report" demo.
"""

from langchain_core.prompts import PromptTemplate
from shared_llm import get_model

template = PromptTemplate(
    template=(
        "Write a short, friendly quote message for a client who wants "
        "'{service}' built. Mention it typically takes about {timeline} "
        "and costs roughly {price_range}. Keep it under 60 words."
    ),
    input_variables=["service", "timeline", "price_range"],
)

# Save this template to disk so it can be reused/version-controlled.
template.save("quote_template.json")

# Load it back (useful once this is a shared asset across projects).
from langchain_core.prompts import load_prompt
loaded_template = load_prompt("quote_template.json")

model = get_model()

filled_prompt = loaded_template.invoke({
    "service": "a WhatsApp order-taking bot",
    "timeline": "7-10 days",
    "price_range": "PKR 40,000-60,000",
})
print(model.invoke(filled_prompt).content)

# Try this next: add a 4th variable ("client_name") and personalize
# the opening line.
