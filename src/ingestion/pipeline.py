# src/ingestion/pipeline.py
from loader import load_documents
from chunker import chunk_documents
from embedder import EmbeddingGenerator
from vector_store import VectorStore

def ingest(folder="data/raw"):
    docs = load_documents(folder)
    chunks = chunk_documents(docs)
    vectors = EmbeddingGenerator().encode_batch([c.page_content for c in chunks])
    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, chunks)
    store.save()
    print(f"{len(docs)} pages → {len(chunks)} chunks indexed")
    return store

if __name__ == "__main__":
    ingest()