"""
Topic: Runnables (LCEL) — RunnableSequence

The | operator builds a RunnableSequence: each step's output becomes
the next step's input. You've already been using this everywhere as
`prompt | model | parser` -- this file shows the explicit, spelled-out
version so you see what the | is actually doing.

Example: service-description -> social post -> hashtag suggestions,
chained explicitly.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from shared_llm import get_model

model = get_model()
parser = StrOutputParser()

post_prompt = PromptTemplate(
    template="Write a 2-sentence Instagram caption promoting this service: {service}",
    input_variables=["service"],
)
hashtag_prompt = PromptTemplate(
    template="Suggest 5 relevant hashtags for this caption:\n\n{caption}",
    input_variables=["caption"],
)

# Explicit form -- functionally identical to:
#   post_prompt | model | parser | hashtag_prompt | model | parser
chain = RunnableSequence(post_prompt, model, parser, hashtag_prompt, model, parser)

result = chain.invoke({"service": "AI-powered inventory tracking for small retail stores"})
print(result)
