from pathlib import Path
import json
import pandas as pd
from scipy import sparse
import joblib
from src.indexer import build_index
from src.searcher import PlotSearcher

def test_basic_search(tmp_path: Path):
    df = pd.DataFrame([
        {"title": "Heist in Boston", "plot": "A daring bank heist in Boston goes wrong."},
        {"title": "Time Travel Tale", "plot": "A scientist invents a time machine to fix the past."},
    ])
    vec, X = build_index(df)
    art = tmp_path / "artifacts"
    art.mkdir()
    joblib.dump(vec, art / "tfidf.joblib")
    sparse.save_npz(art / "matrix.npz", X)
    (art / "meta.json").write_text(json.dumps({"titles": df.title.tolist(), "years": ["",""], "genres": ["",""]}))
    s = PlotSearcher.from_artifacts(art / "tfidf.joblib", art / "matrix.npz", art / "meta.json")
    hits = s.search("bank heist", k=1)
    assert hits and "Heist" in hits[0]["title"]
