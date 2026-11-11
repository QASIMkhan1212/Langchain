"""
Project 6 (chains) — Support ticket router.

Classify an incoming ticket (billing / technical / general), then
CONDITIONALLY chain to a specialized response-generator chain based on
that classification. This is langchain-chains' conditional_chain.py
pattern applied to something you'd actually use for a client project
(you build automation workflows for clients -- this is a drop-in
pattern for a support-inbox bot).
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch

from shared_llm import get_model

model = get_model(temperature=0)
parser = StrOutputParser()


class TicketCategory(BaseModel):
    category: Literal["billing", "technical", "general"]


classify_prompt = PromptTemplate(
    template="Classify this support ticket as billing, technical, or general:\n\n{ticket}",
    input_variables=["ticket"],
)
classify_chain = classify_prompt | model.with_structured_output(TicketCategory)

billing_prompt = PromptTemplate(
    template=(
        "You are a billing support agent. Write a short, empathetic reply to this "
        "billing ticket, and ask for the invoice/transaction ID if not already given:\n\n{ticket}"
    ),
    input_variables=["ticket"],
)
technical_prompt = PromptTemplate(
    template=(
        "You are a technical support agent. Write a short reply to this technical "
        "ticket, asking for 2-3 specific diagnostic details (OS, error message, steps "
        "to reproduce):\n\n{ticket}"
    ),
    input_variables=["ticket"],
)
general_prompt = PromptTemplate(
    template="Write a short, friendly reply acknowledging this general inquiry:\n\n{ticket}",
    input_variables=["ticket"],
)

respond_chain = RunnableBranch(
    (lambda x: x["category"] == "billing", billing_prompt | model | parser),
    (lambda x: x["category"] == "technical", technical_prompt | model | parser),
    general_prompt | model | parser,  # default/fallback
)


def route_ticket(ticket_text: str) -> dict:
    classification = classify_chain.invoke({"ticket": ticket_text})
    reply = respond_chain.invoke({"ticket": ticket_text, "category": classification.category})
    return {"category": classification.category, "reply": reply}


if __name__ == "__main__":
    tickets = [
        "I was charged twice for my subscription this month, please refund the extra charge.",
        "The app crashes every time I try to upload a file larger than 10MB.",
        "Do you offer a student discount?",
    ]
    for t in tickets:
        result = route_ticket(t)
        print(f"\nTicket: {t}")
        print(f"Category: {result['category']}")
        print(f"Reply: {result['reply']}")

# Try this next: add a 4th category "urgent" that skips the queue and
# log all tickets + categories to a CSV (tie-in with langchain-document-loaders).
