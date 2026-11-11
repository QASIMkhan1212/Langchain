"""
Topic: Output Parsers — PydanticOutputParser

Same idea as JsonOutputParser, but parses straight into a validated
Pydantic object instead of a raw dict -- so you get type safety AND
provider-agnostic parsing (no dependency on native structured-output
support) in one step.

Example: parsing a client's budget-negotiation message into a
structured counter-offer record.
"""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from shared_llm import get_model

CLIENT_MESSAGE = (
    "Your quote of PKR 50,000 is a bit high for us. We could do PKR 35,000 "
    "upfront, with the rest after we see the first working version."
)


class CounterOffer(BaseModel):
    original_quote_pkr: int = Field(description="The quote being negotiated, if mentioned")
    proposed_amount_pkr: int
    payment_structure: str = Field(description="How payment would be split, in plain words")


model = get_model(temperature=0)
parser = PydanticOutputParser(pydantic_object=CounterOffer)

prompt = PromptTemplate(
    template="Extract the counter-offer details from this message.\n\n{message}\n\n{format_instructions}",
    input_variables=["message"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser
result = chain.invoke({"message": CLIENT_MESSAGE})
print(result.model_dump_json(indent=2))
