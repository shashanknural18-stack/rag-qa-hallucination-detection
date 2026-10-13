# src/ingestion/vector_store.py
import faiss, json, numpy as np
from pathlib import Path

class VectorStore:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatIP(dim)
        self.records = []   # position i here == vector i in the index

    def add(self, vectors: np.ndarray, chunks):
        self.index.add(vectors)
        self.records.extend(
            {"text": c.page_content, "metadata": c.metadata} for c in chunks
        )

    def search(self, query_vec: np.ndarray, k=10):
        scores, ids = self.index.search(query_vec.reshape(1, -1), k)
        return [(self.records[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, folder="data/index"):
        Path(folder).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{folder}/faiss.index")
        json.dump(self.records, open(f"{folder}/records.json", "w"))

    @classmethod
    def load(cls, folder="data/index"):
        vs = cls()
        vs.index = faiss.read_index(f"{folder}/faiss.index")
        vs.records = json.load(open(f"{folder}/records.json"))
        return vs