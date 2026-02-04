"""
Document ingestion pipeline.

Flow: PDF files → text extraction → chunking → embeddings → Chroma vector store

Key decisions:
- chunk_size=1000: ~1 paragraph. Small enough for precise retrieval,
  large enough to carry meaning.
- chunk_overlap=200: prevents losing context at chunk boundaries.
  If a key sentence spans two chunks, both will contain it.
- Chroma: local vector DB, no server setup needed. Data persists to disk.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHROMA_DIR = "./chroma_db"


def load_pdf(file_path: str) -> list:
    """Load a PDF and return a list of Document objects (one per page)."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def chunk_documents(
    documents: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list:
    """
    Split documents into overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on these separators
    in order: ["\n\n", "\n", " ", ""], preserving paragraph and sentence
    boundaries when possible. This is better than naive fixed-size splits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks: list, collection_name: str = "pdf_docs") -> Chroma:
    """
    Embed chunks and store them in Chroma.

    Each chunk gets converted to a vector via OpenAI embeddings,
    then stored alongside its text and metadata (page number, source file).
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
    )
    return vector_store


def ingest_pdf(file_path: str) -> Chroma:
    """Full pipeline: PDF → chunks → vector store."""
    print(f"Loading PDF: {file_path}")
    documents = load_pdf(file_path)
    print(f"  Loaded {len(documents)} pages")

    chunks = chunk_documents(documents)
    print(f"  Split into {len(chunks)} chunks")

    vector_store = create_vector_store(chunks)
    print(f"  Stored in Chroma at {CHROMA_DIR}")

    return vector_store


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path-to-pdf>")
        sys.exit(1)

    ingest_pdf(sys.argv[1])
