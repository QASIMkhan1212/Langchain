"""
Topic: Runnables (LCEL) — RunnableLambda

Wraps a plain Python function so it can sit inside an LCEL chain like
any other Runnable. Use this whenever a step is just custom logic
(string cleanup, a calculation, a lookup) rather than a model call.

Example: strip pricing-sensitive numbers out of a client message
before it's summarized by the model (a "don't leak the budget into
the wrong downstream step" kind of utility step).
"""

import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from shared_llm import get_model

model = get_model()
parser = StrOutputParser()


def redact_numbers(data: dict) -> dict:
    cleaned = re.sub(r"\d+", "[number]", data["message"])
    return {"message": cleaned}


redact_step = RunnableLambda(redact_numbers)
summarize_prompt = PromptTemplate.from_template("Summarize this client message in one sentence:\n\n{message}")

chain = redact_step | summarize_prompt | model | parser

result = chain.invoke({
    "message": "Budget is $500, need it done in 10 days, and we have 3 branches to cover."
})
print(result)

# Try this next: add a 2nd RunnableLambda step that lowercases the
# summary, to see two RunnableLambdas chained back to back.
