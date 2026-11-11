"""
Topic: Structured Output — Pydantic

Pydantic adds real validation + defaults + nested models on top of
what TypedDict gives you -- this is what you'll actually use for
anything that needs to be reliable (e.g. feeding a downstream system).

Example: parsing a freelance invoice request into a validated object
with a nested LineItem model.
"""

from typing import Optional
from pydantic import BaseModel, Field
from shared_llm import get_model

INVOICE_REQUEST = """
Please invoice Al-Noor Traders for:
- n8n workflow setup: 1 unit, PKR 35,000
- 2 months of maintenance/support, PKR 5,000 each
Due in 15 days. My bank is Meezan, account ending 4417.
"""


class LineItem(BaseModel):
    description: str
    quantity: int = Field(default=1)
    unit_price_pkr: float


class Invoice(BaseModel):
    client_name: str
    line_items: list[LineItem]
    total_pkr: float = Field(description="Sum of all line items * quantity")
    due_in_days: Optional[int] = Field(default=None)
    payment_note: Optional[str] = Field(default=None, description="Bank info or payment instructions, if mentioned")


model = get_model(temperature=0)
structured_model = model.with_structured_output(Invoice)

result = structured_model.invoke(f"Extract invoice details:\n\n{INVOICE_REQUEST}")
print(result.model_dump_json(indent=2))

# Validation in action: try changing `total_pkr: float` to
# `total_pkr: int` and see Pydantic coerce/reject accordingly -- that's
# the real difference from TypedDict.
