# Financial Report Analyzer

A RAG (Retrieval-Augmented Generation) based system for querying financial PDF reports — built as an academic project. Ask natural-language questions about financial filings (e.g. 10-Ks, annual reports) and get answers grounded in the actual document content.

## How It Works

1. **Ingestion** — PDFs are parsed and text extracted using `pdfplumber`
2. **Chunking** — Extracted text is split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`
3. **Embedding & Storage** — Chunks are embedded with `sentence-transformers` (`all-MiniLM-L6-v2`) and stored in a **FAISS** vector index
4. **Retrieval & Generation** — On a user query, relevant chunks are retrieved from FAISS and passed to a locally-run **Ollama** LLM (`llama3.2:3b`) to generate a grounded answer
5. **UI** — A **Streamlit** app ties it all together for interactive querying

## Tech Stack

| Component        | Tool                                  |
|-------------------|----------------------------------------|
| PDF parsing       | pdfplumber                            |
| Text chunking     | LangChain (RecursiveCharacterTextSplitter) |
| Embeddings        | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store      | FAISS                                 |
| LLM               | Ollama (llama3.2:3b) — runs locally   |
| UI                | Streamlit                             |

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed
- ~8GB free RAM recommended (for running llama3.2:3b comfortably)

## Setup

Clone the repo:

```bash
git clone git@github.com:Nitish-Sejwal/Financial-Report-Analyzer.git
cd Financial-Report-Analyzer
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Pull the local LLM model (do this ahead of time — it's a few GB):

```bash
ollama pull llama3.2:3b
```

## Running the App

Make sure Ollama is running:

```bash
ollama serve &
```

Then launch the Streamlit app:

```bash
streamlit run app.py
```

This will open the app in your browser (default: `http://localhost:8501`).

## Project Structure

```
financial-report-analyzer/
├── app.py                  # Streamlit UI entry point
├── data/
│   └── reports/             # Sample financial PDFs (Amazon, 10-K filings)
├── src/
│   ├── __init__.py
│   ├── ingest.py            # PDF ingestion logic
│   ├── chunking.py          # Text chunking logic
│   ├── vector_store.py      # FAISS index build/query
│   └── rag_chain.py         # RAG chain: retrieval + Ollama generation
├── requirements.txt
└── README.md
```

## Usage

1. Launch the app (see above)
2. Upload or select a financial report PDF (sample reports included in `data/reports/`)
3. Ask a question in natural language, e.g.:
   - *"What was the total revenue in the last fiscal year?"*
   - *"Summarize the risk factors mentioned in this filing."*
4. The app retrieves relevant sections from the document and generates an answer using the local LLM

## Notes

- The FAISS index is **not** committed to this repo — it's rebuilt from the source PDFs via the ingestion pipeline. Run the ingestion step on first setup before querying.
- The LLM runs entirely locally via Ollama — no external API calls or API keys required.
- Tested against Apple and Amazon 10-K/annual report filings.

## License

Academic project — for coursework/demonstration purposes.
