"""
Topic: Chains — Parallel chain

Same mechanic as 05-langchain-runnables/runnable_parallel.py, applied
to a chains-specific use case: generating multiple PRICING TIERS for
one service at once, since they all stem from the same base service
description but shouldn't wait on each other.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from shared_llm import get_model

model = get_model()
parser = StrOutputParser()

SERVICE = "automated lead scraping + WhatsApp outreach system"

basic = PromptTemplate.from_template("Describe a BASIC tier for: {service}. One sentence, cheapest option.") | model | parser
standard = PromptTemplate.from_template("Describe a STANDARD tier for: {service}. One sentence, mid-range.") | model | parser
premium = PromptTemplate.from_template("Describe a PREMIUM tier for: {service}. One sentence, full-featured.") | model | parser

pricing_chain = RunnableParallel(basic=basic, standard=standard, premium=premium)

result = pricing_chain.invoke({"service": SERVICE})
for tier, description in result.items():
    print(f"{tier.upper()}: {description}")
