"""
Topic: Runnables (LCEL) — RunnableParallel

Runs multiple Runnables on the SAME input at once, and returns a dict
of their results. Useful whenever you need several independent outputs
from one source instead of running them one after another.

Example: from a single client brief, generate a project title, a
timeline estimate, and a risk list -- all in parallel instead of 3
separate sequential calls.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from shared_llm import get_model

model = get_model()
parser = StrOutputParser()

BRIEF = "Client wants a system that scrapes property listings from 3 real estate sites daily and emails new matches."

title_chain = PromptTemplate.from_template("Give this project a short title:\n\n{brief}") | model | parser
timeline_chain = PromptTemplate.from_template("Estimate a realistic timeline in days for:\n\n{brief}") | model | parser
risk_chain = PromptTemplate.from_template("List 2 technical risks for:\n\n{brief}") | model | parser

parallel_chain = RunnableParallel(
    title=title_chain,
    timeline=timeline_chain,
    risks=risk_chain,
)

result = parallel_chain.invoke({"brief": BRIEF})
print("Title:", result["title"])
print("\nTimeline:", result["timeline"])
print("\nRisks:", result["risks"])

# Try this next: time this against running the 3 chains one after
# another in a loop -- RunnableParallel genuinely runs them
# concurrently, so it should be noticeably faster.
