"""
Topic: Chains — Conditional chain

Classify first, then branch to a different chain based on the
classification -- the full pattern that 02-langchain-prompts/project.py 
06_ticket_router.py builds into a complete project. This file isolates
just the 2-step mechanic: classify -> RunnableBranch.

Example: classify a client message as "ready to proceed" vs "still
has objections", and respond differently.
"""

from typing import Literal
from pydantic import BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from shared_llm import get_model

model = get_model(temperature=0)
parser = StrOutputParser()


class Intent(BaseModel):
    intent: Literal["ready_to_proceed", "has_objections"]


classify_prompt = PromptTemplate.from_template("Classify this client message: {message}")
classify_chain = classify_prompt | model.with_structured_output(Intent)

proceed_prompt = PromptTemplate.from_template("Write a reply confirming next steps + asking for a deposit:\n\n{message}")
objection_prompt = PromptTemplate.from_template("Write a reply addressing their concern directly:\n\n{message}")

respond_chain = RunnableBranch(
    (lambda x: x["intent"] == "ready_to_proceed", proceed_prompt | model | parser),
    objection_prompt | model | parser,
)


def handle_message(message: str) -> str:
    intent = classify_chain.invoke({"message": message})
    return respond_chain.invoke({"message": message, "intent": intent.intent})


if __name__ == "__main__":
    print(handle_message("Okay this looks good, let's move forward."))
    print()
    print(handle_message("I'm worried this will take longer than 10 days."))
