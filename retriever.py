"""
Vector store retrieval interface.

Loads the persisted Chroma store and provides similarity search.
Separated from ingest.py so the app can query without re-embedding.
"""

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_DIR = "./chroma_db"


def get_vector_store(collection_name: str = "pdf_docs") -> Chroma:
    """Load existing Chroma vector store from disk."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def get_retriever(k: int = 4):
    """
    Return a LangChain retriever that fetches the top-k most similar chunks.

    This retriever can be plugged directly into a LangChain chain.
    Under the hood: query → embed → cosine similarity search → top-k docs.
    """
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})
