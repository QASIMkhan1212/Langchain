"""
Topic: Prompts — ChatPromptTemplate + roles

Chat models take a LIST of role-tagged messages (system/human/ai), not
one flat string. ChatPromptTemplate builds that list from a template,
which matters once you need a persona (system role) separate from the
user's actual question.

Example: an intake assistant for a freelance automation studio that
asks clarifying questions about a project request.
"""

from langchain_core.prompts import ChatPromptTemplate
from shared_llm import get_model

chat_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are an intake assistant for {studio_name}, a freelance automation "
     "studio. Ask ONE clarifying question to scope a client's request. "
     "Be concise and professional."),
    ("human", "{client_request}"),
])

model = get_model()

prompt_value = chat_template.invoke({
    "studio_name": "Automation Studio",
    "client_request": "I want something that posts to Instagram automatically every day.",
})

response = model.invoke(prompt_value)
print(response.content)

# Try this next: print prompt_value.to_messages() BEFORE calling the
# model, to see the actual list of SystemMessage/HumanMessage objects
# that gets sent -- that's what "chat template" really produces.
