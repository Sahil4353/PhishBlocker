# scripts/data_prep/build_clean_phishing.py
from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/nazario")
OUT_PATH = Path("data/processed/clean_phishing.parquet")

HEADER_CUTOFF_RE = re.compile(r"\n\n", re.MULTILINE)

def extract_body(text: str) -> str:
    m = HEADER_CUTOFF_RE.split(text, maxsplit=1)
    body = m[1] if len(m) > 1 else m[0]
    return body.strip()

def main():
    rows = []
    seen = set()
    for root, _, files in os.walk(RAW_DIR):
        for fn in files:
            fpath = Path(root) / fn
            try:
                raw = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            body = extract_body(raw)
            if not body or len(body) <= 40:
                continue
            h = hashlib.sha256(body.encode("utf-8", errors="ignore")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            rows.append({"body_text": body, "label": "phishing"})

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[nazario] wrote {OUT_PATH} rows={len(df)}")

if __name__ == "__main__":
    main()