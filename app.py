"""
Streamlit UI for the RAG chatbot.

Flow:
1. User uploads a PDF via the sidebar
2. PDF is ingested (chunked, embedded, stored in Chroma)
3. User asks questions in a chat interface
4. Answers are generated from the RAG chain (grounded in the PDF)
"""

import tempfile
from pathlib import Path

import streamlit as st

from ingest import ingest_pdf
from chain import build_rag_chain

st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="wide")
st.title("PDF Chat — RAG Q&A")

# --- Sidebar: PDF upload ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and "ingested" not in st.session_state:
        with st.spinner("Processing PDF..."):
            # Write uploaded file to a temp path so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            vector_store = ingest_pdf(tmp_path)
            st.session_state["ingested"] = True
            Path(tmp_path).unlink()  # clean up temp file

        st.success(f"Ingested: {uploaded_file.name}")

    if "ingested" in st.session_state:
        if st.button("Clear and upload new PDF"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# --- Chat interface ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about your PDF"):
    if "ingested" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        # Show user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer — LCEL chain returns a plain string
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = build_rag_chain()
                answer = chain.invoke(prompt)
                st.markdown(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})
