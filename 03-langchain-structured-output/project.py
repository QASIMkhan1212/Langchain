"""
Project 3 (structured-output) — Resume / CV parser.

Builds on langchain-structured-output: instead of a product review,
extract structured fields from a resume using with_structured_output().
Works with Groq and Gemini (both support tool-calling / JSON mode,
which is what with_structured_output relies on under the hood).
"""

from typing import Optional
from pydantic import BaseModel, Field

from shared_llm import get_model

SAMPLE_RESUME = """
Ayesha Raza
Email: ayesha.raza@example.com | Karachi, Pakistan

Summary: Frontend developer with 3 years of experience building React
applications, currently exploring backend and ML.

Experience:
- Frontend Developer, PixelWorks (2022-Present): Built and maintained
  a React/TypeScript dashboard used by 50k+ monthly users.
- Junior Developer, CodeNest (2021-2022): Shipped internal tooling in
  Vue.js and Node.

Skills: React, TypeScript, Vue, Node.js, Git, basic Python

Education: BS Computer Science, FAST-NUCES (2017-2021)
"""


class Experience(BaseModel):
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    duration: str = Field(description="How long they worked there")


class Resume(BaseModel):
    name: str
    email: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None)
    years_experience: Optional[int] = Field(
        default=None, description="Total years of professional experience, estimated if not stated"
    )
    skills: list[str] = Field(description="All technical skills mentioned")
    experience: list[Experience]
    education: Optional[str] = Field(default=None)


model = get_model(temperature=0)
structured_model = model.with_structured_output(Resume)

result = structured_model.invoke(f"Extract structured data from this resume:\n\n{SAMPLE_RESUME}")

print(result.model_dump_json(indent=2))

# Try this next: swap SAMPLE_RESUME for a PDF loaded via
# 07-langchain-document-loaders/pdf_loader.py and run this on a real resume.
