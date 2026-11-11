"""
Step 1 of the capstone — ingest.

load (document-loaders) -> split (text-splitters) -> embed -> store in
FAISS, saved to disk so agent.py doesn't re-embed every run.

Run this once before agent.py. Re-run it any time you add new docs to
sample_docs/.
"""

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared_llm import get_embeddings

PDF_PATH = "sample_docs/sample.pdf"
WEB_URL = "https://python.langchain.com/docs/concepts/document_loaders/"
INDEX_DIR = "faiss_index"


def load_documents() -> list:
    documents = []

    pdf_docs = PyPDFLoader(PDF_PATH).load()
    for d in pdf_docs:
        d.metadata["source_type"] = "pdf"
    documents.extend(pdf_docs)
    print(f"Loaded {len(pdf_docs)} page(s) from PDF")

    try:
        web_docs = WebBaseLoader(WEB_URL).load()
        for d in web_docs:
            d.metadata["source_type"] = "web"
        documents.extend(web_docs)
        print(f"Loaded {len(web_docs)} page(s) from web")
    except Exception as e:
        print(f"Web load skipped: {e}")

    return documents


def build_index():
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"Saved index to ./{INDEX_DIR}/")


if __name__ == "__main__":
    build_index()

# To use your OWN client's documents instead: drop PDFs into
# sample_docs/, add a DirectoryLoader call here, and re-run.
