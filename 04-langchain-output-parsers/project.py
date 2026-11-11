"""
Project 4 (output-parsers) — News headline -> structured JSON.

Unlike Project 3, this deliberately uses a model/parser combo where
you control the output format via PROMPT INSTRUCTIONS + a parser,
rather than with_structured_output(). That's the actual skill the
output-parsers repo teaches: forcing structure out of a model that
has no native structured-output mode, or when you want to see/debug
the raw JSON the model produces.
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from shared_llm import get_model

HEADLINES = [
    "Tech layoffs continue as three more startups announce job cuts this week",
    "Local team wins championship after dramatic last-minute comeback",
    "New study finds moderate exercise improves sleep quality in adults",
]


class HeadlineAnalysis(BaseModel):
    category: str = Field(description="One of: technology, sports, health, politics, business, other")
    sentiment: str = Field(description="One of: positive, negative, neutral")
    one_line_summary: str = Field(description="A single, punchy sentence summarizing the headline")


model = get_model(temperature=0)
parser = JsonOutputParser(pydantic_object=HeadlineAnalysis)

prompt = PromptTemplate(
    template=(
        "Analyze this news headline.\n\nHeadline: {headline}\n\n"
        "{format_instructions}"
    ),
    input_variables=["headline"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

for headline in HEADLINES:
    result = chain.invoke({"headline": headline})
    print(f"\nHeadline: {headline}")
    print(result)

# Try this next: swap JsonOutputParser for PydanticOutputParser and
# compare the format_instructions text each one generates -- that's
# what actually goes into the prompt the model sees.
