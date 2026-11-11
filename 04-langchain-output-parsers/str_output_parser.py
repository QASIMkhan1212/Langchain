"""
Topic: Output Parsers — StrOutputParser

The simplest parser: takes the model's AIMessage object and pulls out
just the .content string, so a chain's output is a plain string
instead of a message object. Mostly useful for chaining steps where
you don't need the extra message metadata.

Example: a 2-step chain -- generate a cold outreach opener, then
shorten it -- showing why StrOutputParser matters for chaining.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from shared_llm import get_model

model = get_model()
parser = StrOutputParser()

opener_prompt = PromptTemplate(
    template="Write a 2-sentence cold outreach opener to a {business_type} owner about automating {pain_point}.",
    input_variables=["business_type", "pain_point"],
)
shorten_prompt = PromptTemplate(
    template="Shorten this to a single punchy sentence, under 20 words:\n\n{text}",
    input_variables=["text"],
)

# Without StrOutputParser, step 2's {text} would need to be an
# AIMessage, and PromptTemplate expects a plain string -- this is the
# glue that makes chaining steps actually work.
chain = opener_prompt | model | parser | shorten_prompt | model | parser

result = chain.invoke({"business_type": "local pharmacy", "pain_point": "inventory tracking"})
print(result)
