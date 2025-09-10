from __future__ import annotations
import argparse
from pathlib import Path
import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import load_movies_csv, save_json

def build_index(df, max_features=50000, ngram_range=(1,2)):
    corpus = df["plot"].astype(str).tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="artifacts/tfidf.joblib")
    p.add_argument("--matrix", default="artifacts/matrix.npz")
    p.add_argument("--meta", default="artifacts/meta.json")
    ns = p.parse_args(args)

    df = load_movies_csv(ns.data)
    vec, X = build_index(df)

    Path(ns.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, ns.model)
    sparse.save_npz(ns.matrix, X)

    meta = {
        "titles": df["title"].astype(str).tolist(),
        "years": df["year"].astype(str).tolist() if "year" in df else [],
        "genres": df["genres"].astype(str).tolist() if "genres" in df else [],
    }
    save_json(meta, ns.meta)
    print(f"Saved: {ns.model}, {ns.matrix}, {ns.meta}")

if __name__ == "__main__":
    main()
