# scripts/data_prep/build_clean_phishing.py
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/nazario")
OUT_PATH = Path("data/processed/clean_phishing.csv")

HEADER_CUTOFF_RE = re.compile(r"\n\n", re.MULTILINE)


def extract_body(text: str) -> str:
    """Drop headers before the first blank line and keep the body."""
    parts = HEADER_CUTOFF_RE.split(text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]
    return body.strip()


def main():
    rows: list[dict[str, str]] = []

    for root, _, files in os.walk(RAW_DIR):
        for fn in files:
            fpath = Path(root) / fn
            try:
                raw = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                # unreadable file → skip
                continue
            body = extract_body(raw)
            # Keep everything, even short ones
            rows.append({"body_text": body, "label": "phishing"})

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[nazario] wrote {OUT_PATH} rows={len(df)} (all labeled 'phishing')")


if __name__ == "__main__":
    main()
