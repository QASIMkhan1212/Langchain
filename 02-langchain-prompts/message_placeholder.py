"""
Topic: Prompts — MessagesPlaceholder (dynamic chat history)

ChatPromptTemplate's slots are usually fixed (one system message, one
human message). MessagesPlaceholder is a slot for a VARIABLE-LENGTH
list of past messages -- this is how multi-turn memory gets inserted
into a prompt. (The full working chatbot using this lives in
02-langchain-prompts/project.py -- this file isolates
just the concept.)
"""

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from shared_llm import get_model

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a support agent for an automation studio. Be concise."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# Simulated past conversation -- in a real app this comes from a
# database, a file, or session state, not a hardcoded list like this.
history = [
    HumanMessage(content="I set up the n8n workflow but it's not triggering."),
    AIMessage(content="Got it -- is the webhook node active, and is the workflow itself toggled 'active' in n8n?"),
]

model = get_model()

prompt_value = chat_template.invoke({
    "history": history,
    "question": "Yes it's active, but I'm still not seeing any executions.",
})

response = model.invoke(prompt_value)
print(response.content)

# Try this next: print len(prompt_value.to_messages()) -- it's 4, not
# 2 -- the system message and new question get added around whatever
# history you pass in.
