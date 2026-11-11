"""
Topic: Structured Output — TypedDict

TypedDict is the lightest way to get structured output: no validation,
just a type hint describing the shape you want. Good for quick
prototyping; no runtime checks though (see pydantic_demo.py for that).

Example: extracting a quick lead-qualification snapshot from a raw
client message -- not the usual "movie review" demo.
"""

from typing import TypedDict, Annotated
from shared_llm import get_model

CLIENT_MESSAGE = (
    "Hey, I run a 3-branch clothing store in Lahore and I'm losing track "
    "of inventory across locations. Need this sorted before Eid season, "
    "budget is flexible if it actually works."
)


class LeadSnapshot(TypedDict):
    industry: Annotated[str, "The client's business industry"]
    pain_point: Annotated[str, "The core problem they described"]
    urgency: Annotated[str, "One of: low, medium, high"]
    budget_signal: Annotated[str, "What they implied about budget"]


model = get_model(temperature=0)
structured_model = model.with_structured_output(LeadSnapshot)

result = structured_model.invoke(f"Extract a lead snapshot from this message:\n\n{CLIENT_MESSAGE}")
print(result)

# Try this next: feed it a message with NO clear urgency or budget
# signal and see what it does -- TypedDict gives no fallback/default
# behavior, so watch how the model handles missing info on its own.
