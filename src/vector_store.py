from sentence_transformers import SentenceTransformer

_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts):
    embedder = get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True)
    return vectors

import faiss
import numpy as np
import pickle
from pathlib import Path

def build_index(chunks, index_dir="outputs/faiss_index"):
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    vectors = np.array(vectors, dtype="float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, f"{index_dir}/index.faiss")
    with open(f"{index_dir}/metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved index with {index.ntotal} vectors")


def load_index(index_dir="outputs/faiss_index"):
    index = faiss.read_index(f"{index_dir}/index.faiss")
    with open(f"{index_dir}/metadata.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def search(query, top_k=5):
    index, chunks = load_index()
    query_vec = embed_texts([query])
    query_vec = np.array(query_vec, dtype="float32")

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)
    return results
