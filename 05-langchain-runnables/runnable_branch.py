"""
Topic: Runnables (LCEL) — RunnableBranch

Picks between different Runnables based on a condition -- the LCEL
equivalent of an if/elif/else. Each entry is (condition_fn, runnable);
the last unpaired runnable is the default/fallback.

Example: route a client's budget statement to a different reply style
depending on whether it's above or below your minimum rate.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from shared_llm import get_model

model = get_model()
parser = StrOutputParser()

MIN_BUDGET_USD = 200

accept_prompt = PromptTemplate.from_template(
    "Write a short, enthusiastic reply accepting this budget: ${budget}. Propose next steps."
)
decline_prompt = PromptTemplate.from_template(
    "Write a short, polite reply explaining ${budget} is below the minimum for custom automation work, "
    "and suggest a cheaper pre-built alternative instead."
)

branch = RunnableBranch(
    (lambda x: x["budget"] >= MIN_BUDGET_USD, accept_prompt | model | parser),
    decline_prompt | model | parser,  # default when condition is False
)

for budget in [350, 80]:
    result = branch.invoke({"budget": budget})
    print(f"\n--- budget=${budget} ---")
    print(result)

# Try this next: add a 2nd condition tier for budgets exactly at
# MIN_BUDGET_USD that asks a clarifying question instead of a flat
# accept/decline.
