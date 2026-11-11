"""
Project 7 (document-loaders) — Multi-source document collector.

Loads a local PDF + a live webpage (reusing this folder's sample_data/) and normalizes both into one flat
list of LangChain Document objects. This is the real-world shape of
"prep data for RAG": you almost never have just one source type.
"""

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

PDF_PATH = "sample_data/proposal_template.pdf"
WEB_URL = "https://python.langchain.com/docs/concepts/document_loaders/"


def load_all() -> list:
    documents = []

    pdf_docs = PyPDFLoader(PDF_PATH).load()
    for d in pdf_docs:
        d.metadata["source_type"] = "pdf"
    documents.extend(pdf_docs)
    print(f"Loaded {len(pdf_docs)} page(s) from PDF: {PDF_PATH}")

    try:
        web_docs = WebBaseLoader(WEB_URL).load()
        for d in web_docs:
            d.metadata["source_type"] = "web"
        documents.extend(web_docs)
        print(f"Loaded {len(web_docs)} page(s) from web: {WEB_URL}")
    except Exception as e:
        print(f"Web load skipped (no internet in this environment?): {e}")

    return documents


if __name__ == "__main__":
    docs = load_all()
    print(f"\nTotal documents: {len(docs)}")
    for i, d in enumerate(docs):
        preview = d.page_content[:150].replace("\n", " ")
        print(f"\n[{i}] source_type={d.metadata.get('source_type')}")
        print(f"    {preview}...")

# Try this next: add a DirectoryLoader pointed at a folder of your own
# client project PDFs/notes and merge those in too.
