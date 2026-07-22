import streamlit as st
from pathlib import Path
from src.ingest import load_all_reports
from src.chunking import chunk_documents
from src.vector_store import build_index
from src.rag_chain import ask

st.set_page_config(page_title="Financial Report Analyzer", page_icon="📊")
st.title("📊 Financial Report Analyzer")
st.caption("Ask questions about your indexed financial reports — powered by a local LLM")

DATA_DIR = Path("data/reports")

# --- Sidebar: upload + rebuild ---
with st.sidebar:
    st.header("1. Upload Reports")
    uploaded_files = st.file_uploader("Upload PDF financial reports", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded_files:
            dest = DATA_DIR / f.name
            dest.write_bytes(f.read())
        st.success(f"Saved {len(uploaded_files)} file(s)")

    st.header("2. Build Index")
    if st.button("🔄 Rebuild Index", use_container_width=True):
        with st.spinner("Extracting, chunking, and embedding..."):
            docs = load_all_reports()
            if not docs:
                st.error("No PDFs found in data/reports/")
            else:
                chunks = chunk_documents(docs)
                build_index(chunks)
                st.success(f"Indexed {len(chunks)} chunks from {len(docs)} document(s)")

    st.divider()
    existing = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    st.write(f"**Reports on disk:** {len(existing)}")
    for p in existing:
        st.write(f"- {p.name}")

# --- Main: Q&A ---
question = st.text_input("Ask a question about your financial reports:")

if question:
    with st.spinner("Retrieving context and generating answer..."):
        result = ask(question)

    st.markdown("### Answer")
    st.write(result["answer"])

    with st.expander("📁 Sources used"):
        for s in result["sources"]:
            st.write(f"**{s['source']}** — chunk {s['chunk_index']} (relevance: {s['score']:.3f})")
