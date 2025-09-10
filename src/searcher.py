from __future__ import annotations
import numpy as np
from scipy import sparse
import joblib
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import load_json

class PlotSearcher:
    def __init__(self, vectorizer: TfidfVectorizer, matrix, meta):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.meta = meta

    @classmethod
    def from_artifacts(cls, model_path, matrix_path, meta_path):
        vec = joblib.load(model_path)
        X = sparse.load_npz(matrix_path)
        meta = load_json(meta_path)
        return cls(vec, X, meta)

    def search(self, query: str, k: int = 10):
        q = self.vectorizer.transform([query])
        sims = linear_kernel(q, self.matrix).ravel()
        idx = np.argsort(-sims)[:k]
        results = []
        for rank, i in enumerate(idx, 1):
            results.append({
                "rank": rank,
                "title": self.meta["titles"][i],
                "year": self.meta["years"][i] if self.meta.get("years") else "",
                "genres": self.meta["genres"][i] if self.meta.get("genres") else "",
                "score": float(sims[i]),
            })
        return results
