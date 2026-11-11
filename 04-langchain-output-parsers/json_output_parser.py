"""
Topic: Output Parsers — JsonOutputParser

Forces the model to return parseable JSON by injecting format
instructions into the prompt, then parses the raw text into a Python
dict. Different from with_structured_output() (topic 3) because it
works even with models/providers that don't have a native
structured-output mode -- it's prompt-engineering the structure in,
not API-enforcing it.

Example: scoring a freelance gig posting for fit.
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from shared_llm import get_model

GIG_POSTING = (
    "Need an automation expert to build a WhatsApp bot that answers FAQs "
    "and forwards complex queries to a human. Budget: $150. Timeline: ASAP."
)

model = get_model(temperature=0)
parser = JsonOutputParser()

prompt = PromptTemplate(
    template=(
        "Score this freelance gig posting for a developer who specializes in "
        "WhatsApp/n8n automation. Return JSON with keys: fit_score (1-10), "
        "red_flags (list of strings), recommended_rate_usd (number).\n\n"
        "Posting: {posting}\n\n{format_instructions}"
    ),
    input_variables=["posting"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser
result = chain.invoke({"posting": GIG_POSTING})
print(result)

# Try this next: print parser.get_format_instructions() on its own --
# that's the literal text getting injected into the prompt to coax JSON
# out of the model.
