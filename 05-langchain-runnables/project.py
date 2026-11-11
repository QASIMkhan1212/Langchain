"""
Project 5 (runnables / LCEL) — Content generation pipeline.

Topic -> blog post -> IN PARALLEL: SEO title + one-line summary
       -> BRANCH on word count: long posts get a "Read time" note,
          short posts get a "needs expansion" note.

Exercises RunnableSequence (the | chain), RunnableParallel (the two
simultaneous sub-tasks), and RunnableBranch (the word-count decision)
all in one pipeline -- which is exactly what langchain-runnables covers.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel

from shared_llm import get_model

model = get_model()
parser = StrOutputParser()

# --- Step 1: generate the blog post ---
blog_prompt = PromptTemplate(
    template="Write a short blog post (150-250 words) about: {topic}",
    input_variables=["topic"],
)
generate_post = blog_prompt | model | parser

# --- Step 2: in parallel, generate an SEO title and a one-line summary ---
title_prompt = PromptTemplate(
    template="Write one punchy, SEO-friendly title (under 10 words) for this post:\n\n{post}",
    input_variables=["post"],
)
summary_prompt = PromptTemplate(
    template="Summarize this post in exactly one sentence:\n\n{post}",
    input_variables=["post"],
)

enrich = RunnableParallel(
    post=lambda x: x,
    title=title_prompt | model | parser,
    summary=summary_prompt | model | parser,
)

# --- Step 3: branch on word count ---
def word_count(data: dict) -> int:
    return len(data["post"].split())

add_note = RunnableBranch(
    (
        lambda data: word_count(data) > 180,
        RunnableLambda(lambda data: {**data, "note": f"~{word_count(data)} words, ~1 min read"}),
    ),
    RunnableLambda(lambda data: {**data, "note": f"~{word_count(data)} words, consider expanding"}),
)

pipeline = generate_post | enrich | add_note

if __name__ == "__main__":
    result = pipeline.invoke({"topic": "why Pakistani freelancers should learn agentic AI"})

    print("TITLE:", result["title"])
    print("\nSUMMARY:", result["summary"])
    print("\nNOTE:", result["note"])
    print("\nFULL POST:\n", result["post"])

# Try this next: add a 3rd parallel branch that generates 3 hashtags,
# and a 2nd branch condition for posts under 50 words ("too short, discard").
