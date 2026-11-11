"""
Topic: Text Splitters — Code-aware splitting

Same from_language() mechanism as markdown_splitting.py, but for
source code -- it tries to keep function/class definitions intact
instead of splitting through the middle of one. Useful if you ever
build a RAG system over your own n8n/Python project's codebase.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

PYTHON_CODE = '''
def classify_lead(message: str) -> str:
    """Return 'hot', 'warm', or 'cold' based on urgency signals."""
    if "asap" in message.lower() or "urgent" in message.lower():
        return "hot"
    if "budget" in message.lower():
        return "warm"
    return "cold"


class LeadTracker:
    def __init__(self):
        self.leads = []

    def add(self, message: str):
        category = classify_lead(message)
        self.leads.append((message, category))
'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=150,
    chunk_overlap=0,
)
chunks = splitter.split_text(PYTHON_CODE)

print(f"{len(chunks)} chunks:\n")
for i, c in enumerate(chunks):
    print(f"[{i}]\n{c}\n---")
