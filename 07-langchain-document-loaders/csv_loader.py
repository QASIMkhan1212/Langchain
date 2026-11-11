"""
Topic: Document Loaders — CSVLoader

Loads a CSV where EACH ROW becomes its own Document -- useful when
each row is a self-contained unit of meaning (here: one client lead
per row), unlike a text file where the whole file is usually one unit.
"""

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="sample_data/leads.csv")
docs = loader.load()

print(f"Loaded {len(docs)} document(s) — one per CSV row\n")
for d in docs:
    print(d.page_content)
    print("---")

# Try this next: pass source_column="name" to CSVLoader and check
# docs[0].metadata["source"] -- it'll use that column's value instead
# of the row number.
