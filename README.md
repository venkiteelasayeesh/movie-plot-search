# Movie Plot Search Engine

TF‑IDF–based semantic keyword search over movie plot summaries.With the help of this project ;we can accurately perform query-summary matching.
Includes a FastAPI endpoint and a simple Python interface.


![build](https://img.shields.io/badge/build-passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-informational)


## ✨ Features
- TF‑IDF vectorization with smart text pre‑processing
- Cosine similarity ranking
- FastAPI endpoint: `GET /search?q=...&k=10`
- Lightweight, dependency‑minimal
- Tested with `pytest`


## 📂 Data
Place a CSV at `data/movies.csv` with columns:
- `title` (str)
- `plot` (str)
- `year` (int, optional)
- `genres` (str, optional; comma‑separated)


> Tip: If your source has different names, adjust `utils.py` mapping.


## 🚀 Quickstart
```bash
python -m venv .venv && source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt


# build index
python -m src.indexer --data data/movies.csv --model artifacts/tfidf.joblib --matrix artifacts/matrix.npz --meta artifacts/meta.json


# run API
uvicorn api.main:app --reload --port 8000
# then open: http://127.0.0.1:8000/search?q=time%20travel&k=5
