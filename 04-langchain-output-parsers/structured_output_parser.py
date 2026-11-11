"""
Topic: Output Parsers — StructuredOutputParser (legacy, know it, skip it)

Older than PydanticOutputParser -- you define fields as ResponseSchema
objects instead of a Pydantic class. No validation, no nesting. Shown
here so you recognize it in older tutorials/codebases, but
with_structured_output() (topic 3) or PydanticOutputParser (this
folder) are what you should actually reach for today.

Example: pulling 2 quick fields out of a one-line client reply.
"""

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from shared_llm import get_model

CLIENT_REPLY = "Sounds good, let's start next Monday. I'll pay half now, half on delivery."

response_schemas = [
    ResponseSchema(name="start_date", description="When the client wants to start, in plain words"),
    ResponseSchema(name="payment_terms", description="How they want to pay"),
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt = PromptTemplate(
    template="Extract the following from this message.\n\n{message}\n\n{format_instructions}",
    input_variables=["message"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = get_model(temperature=0)
chain = prompt | model | parser
result = chain.invoke({"message": CLIENT_REPLY})
print(result)
