# scripts/data_prep/build_clean_spam.py
from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path
import pandas as pd

RAW_BASE = Path("data/raw/spamassassin")
OUT_PATH = Path("data/processed/clean_spamassassin.parquet")

HEADER_CUTOFF_RE = re.compile(r"\n\n", re.MULTILINE)

def extract_body(text: str) -> str:
    m = HEADER_CUTOFF_RE.split(text, maxsplit=1)
    body = m[1] if len(m) > 1 else m[0]
    return body.strip()

def load_dir(dir_path: Path, label: str):
    rows = []
    for root, _, files in os.walk(dir_path):
        for fn in files:
            fpath = Path(root) / fn
            try:
                raw = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            body = extract_body(raw)
            if body and len(body) > 40:
                rows.append({"body_text": body, "label": label})
    return rows

def main():
    rows = []
    rows += load_dir(RAW_BASE / "spam", "spam")
    rows += load_dir(RAW_BASE / "easy_ham", "safe")
    rows += load_dir(RAW_BASE / "hard_ham", "safe")

    # dedupe by body hash across both ham+spam so we don't get contradictory labels
    dedup = {}
    for r in rows:
        h = hashlib.sha256(r["body_text"].encode("utf-8", errors="ignore")).hexdigest()
        # conflict resolution: prefer "spam" over "safe" if same text shows up
        prev = dedup.get(h)
        if prev is None:
            dedup[h] = r
        else:
            if prev["label"] != r["label"]:
                # If same content appears labeled safe+spam, lean "spam"
                if "spam" in (prev["label"], r["label"]):
                    dedup[h] = {"body_text": r["body_text"], "label": "spam"}

    df = pd.DataFrame(dedup.values())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[spamassassin] wrote {OUT_PATH} rows={len(df)}")

if __name__ == "__main__":
    main()
