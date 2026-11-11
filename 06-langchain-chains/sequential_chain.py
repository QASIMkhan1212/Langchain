"""
Topic: Chains — Sequential chain

Multiple steps, each depending on the previous one's output, chained
with |. Different from the single-step simple_chain.py: here the
model's own output becomes the next prompt's input, 2 steps deep.

Example: idea -> expanded feature list -> effort estimate, where the
estimate genuinely depends on what the feature list came out to be.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from shared_llm import get_model

model = get_model()
parser = StrOutputParser()

features_prompt = PromptTemplate.from_template(
    "List 4 concrete features for this automation idea: {idea}"
)
estimate_prompt = PromptTemplate.from_template(
    "Given this feature list, estimate total build time in days:\n\n{features}"
)

chain = features_prompt | model | parser | (lambda features: {"features": features}) | estimate_prompt | model | parser

result = chain.invoke({"idea": "automated WhatsApp order bot for a small restaurant"})
print(result)

# Try this next: print the chain's intermediate output by splitting it
# into two separate chains and calling .invoke() on each manually --
# useful for debugging which step a bad result came from.
