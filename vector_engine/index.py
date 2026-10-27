import numpy as np
import pickle
import os
from typing import List, Dict, Any

class VectorIndex:
    def __init__(self, dimension: int = 384, storage_path: str = "vector_store.pkl"):
        self.dimension = dimension
        self.storage_path = storage_path
        self.documents: List[Dict[str, Any]] = []
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self.load()

    def add_documents(self, docs: List[Dict[str, Any]], embeddings: np.ndarray):
        if len(docs) != len(embeddings):
            raise ValueError("Document count must match embedding matrix rows.")

        self.documents.extend(docs)
        if self.vectors.shape[0] == 0:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])
        self.save()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.vectors.shape[0] == 0:
            return []

        if query_vector.ndim > 1:
            query_vector = query_vector.squeeze(0)

        scores = np.dot(self.vectors, query_vector)

        k = min(top_k, len(scores))
        if k == len(scores):
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            results.append({
                "id": self.documents[idx].get("id"),
                "text": self.documents[idx].get("text"),
                "metadata": self.documents[idx].get("metadata", {}),
                "similarity_score": float(scores[idx])
            })
        return results

    def save(self):
        with open(self.storage_path, "wb") as f:
            pickle.dump({"documents": self.documents, "vectors": self.vectors}, f)

    def load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data.get("documents", [])
                self.vectors = data.get("vectors", np.empty((0, self.dimension), dtype=np.float32))

    def count(self) -> int:
        return len(self.documents)
