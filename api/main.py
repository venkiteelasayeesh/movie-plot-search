from __future__ import annotations
from fastapi import FastAPI, Query
from pydantic import BaseModel
from pathlib import Path

# ⬇️ notice we import from src, not api
from src.searcher import PlotSearcher

app = FastAPI(title="Movie Plot Search API")

ARTIFACT_DIR = Path("artifacts")
MODEL = ARTIFACT_DIR / "tfidf.joblib"
MATRIX = ARTIFACT_DIR / "matrix.npz"
META = ARTIFACT_DIR / "meta.json"

class Hit(BaseModel):
    rank: int
    title: str
    year: str | None = None
    genres: str | None = None
    score: float

@app.on_event("startup")
def _load():
    if not (MODEL.exists() and MATRIX.exists() and META.exists()):
        raise RuntimeError("Artifacts missing. Run the indexer first.")
    global searcher
    searcher = PlotSearcher.from_artifacts(MODEL, MATRIX, META)

@app.get("/search", response_model=list[Hit])
def search(q: str = Query(..., min_length=2), k: int = 10):
    return searcher.search(q, k)
