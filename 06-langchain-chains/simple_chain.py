"""
Topic: Chains — Simple chain

A "chain" is just an LCEL sequence (prompt | model | parser) used for
a single, specific task -- same mechanics as 05-langchain-runnables,
different framing: here the focus is "this IS the deliverable", not
"here's how piping works".

Example: turn a rough service idea into a one-paragraph client pitch.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from shared_llm import get_model

model = get_model()
parser = StrOutputParser()

pitch_prompt = PromptTemplate(
    template="Turn this rough idea into a polished 2-sentence client pitch: {idea}",
    input_variables=["idea"],
)

chain = pitch_prompt | model | parser

result = chain.invoke({"idea": "bot that auto-replies to Instagram DMs with product prices"})
print(result)
