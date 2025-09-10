from __future__ import annotations
import json
import pandas as pd
from pathlib import Path

REQUIRED_COLS = {"title", "plot"}

def load_movies_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return df.fillna("")

def save_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
