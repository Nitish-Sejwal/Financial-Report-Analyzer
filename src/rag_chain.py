import requests
from src.vector_store import search

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

SYSTEM_PROMPT = """You are a financial report analyst. Answer the user's question
using ONLY the context provided below. If the context doesn't contain enough
information to answer, say so clearly instead of guessing. Cite which source
chunk your answer comes from when possible."""

def build_context_block(chunks):
    blocks = []
    for c in chunks:
        blocks.append(
            f"--- Source: {c['source']} (chunk {c['chunk_index']}, relevance {c['score']:.3f}) ---\n"
            f"{c['text']}"
        )
    return "\n\n".join(blocks)


def ask(question, top_k=5):
    retrieved = search(question, top_k=top_k)
    context_block = build_context_block(retrieved)

    full_prompt = f"""{SYSTEM_PROMPT}

Context from financial reports:

{context_block}

Question: {question}"""

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    })

    result = response.json()

    return {
        "answer": result["response"],
        "sources": [
            {"source": c["source"], "chunk_index": c["chunk_index"], "score": c["score"]}
            for c in retrieved
        ]
    }
