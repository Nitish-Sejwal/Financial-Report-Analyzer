from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"{doc['source']}::chunk_{i}",
                "text": piece,
                "source": doc["source"],
                "chunk_index": i,
            })

    return chunks
