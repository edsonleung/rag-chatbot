"""
RAG chain: retriever + LLM with a grounded prompt.

Architecture:
    User question
        → retriever fetches top-k chunks
        → chunks are "stuffed" into the prompt as context
        → LLM generates an answer grounded in that context

Uses LCEL (LangChain Expression Language) — the modern way to compose
chains in LangChain v1+. Chains are built with the | pipe operator,
similar to Unix pipes: each step's output feeds into the next.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from retriever import get_retriever

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided "
    "document context. Use ONLY the following context to answer the question. "
    "If the context does not contain enough information to answer, say "
    "'I don't have enough information in the documents to answer that.'\n\n"
    "Context:\n{context}"
)


def _format_docs(docs: list) -> str:
    """Join retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    model: str = "meta-llama/llama-3.3-70b-instruct:free",
    temperature: float = 0.0,
):
    """
    Build and return the full RAG chain using LCEL.

    The chain:
      1. Takes {"input": "user question"}
      2. Retriever fetches top-k docs
      3. Docs are formatted into a context string
      4. Prompt is filled with context + question
      5. LLM generates the answer
      6. StrOutputParser extracts the text

    RunnablePassthrough passes the input through unchanged,
    so "input" is available at every step.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=OPENROUTER_BASE_URL,
    )
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    # LCEL chain: | pipes output of each step to the next
    chain = (
        {
            "context": retriever | _format_docs,  # question → docs → string
            "input": RunnablePassthrough(),        # question passes through
        }
        | prompt    # fills {context} and {input} into the template
        | llm       # generates response
        | StrOutputParser()  # extracts string from AIMessage
    )

    return chain


def ask(question: str) -> str:
    """Ask a question and return the answer string."""
    chain = build_rag_chain()
    return chain.invoke(question)
