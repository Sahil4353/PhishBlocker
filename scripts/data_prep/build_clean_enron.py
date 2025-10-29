# scripts/data_prep/build_clean_enron.py
from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/enron")
OUT_PATH = Path("data/processed/clean_enron.parquet")

HEADER_CUTOFF_RE = re.compile(r"\n\n", re.MULTILINE)  # split headers/body at first blank line

def extract_body(text: str) -> str:
    # crude: drop headers before first blank line
    m = HEADER_CUTOFF_RE.split(text, maxsplit=1)
    body = m[1] if len(m) > 1 else m[0]

    # drop obvious legal footers / boilerplate patterns later if needed
    body = body.strip()
    return body

def walk_enron_texts(base: Path):
    for root, _, files in os.walk(base):
        for fn in files:
            fpath = Path(root) / fn
            try:
                raw = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            body = extract_body(raw)
            if body:
                yield body

def main():
    rows = []
    seen = set()

    for body in walk_enron_texts(RAW_DIR):
        # dedupe aggressively by hash
        h = hashlib.sha256(body.encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rows.append({"body_text": body, "label": "safe"})

    df = pd.DataFrame(rows)
    print(f"[enron] collected={len(df)} unique_safe_emails")

    # drop super-short garbage
    df = df[df["body_text"].str.len() > 40].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[enron] wrote {OUT_PATH} rows={len(df)}")

if __name__ == "__main__":
    main()
