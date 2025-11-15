# scripts/data_prep/build_clean_enron.py
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/enron")
OUT_PATH = Path("data/processed/clean_enron.csv")

# Split headers/body at first blank line
HEADER_CUTOFF_RE = re.compile(r"\n\n", re.MULTILINE)


def extract_body(text: str) -> str:
    """Drop headers before the first blank line and keep the body."""
    parts = HEADER_CUTOFF_RE.split(text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]
    return body.strip()


def walk_enron_texts(base: Path):
    """Yield body text for every readable file under base."""
    for root, _, files in os.walk(base):
        for fn in files:
            fpath = Path(root) / fn
            try:
                raw = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                # unreadable file → skip
                continue
            body = extract_body(raw)
            # Keep everything, even if empty/short
            yield body


def main():
    rows: list[dict[str, str]] = []

    for body in walk_enron_texts(RAW_DIR):
        rows.append({"body_text": body, "label": "safe"})

    df = pd.DataFrame(rows)
    print(f"[enron] collected={len(df)} emails (all labeled 'safe')")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[enron] wrote {OUT_PATH} rows={len(df)}")


if __name__ == "__main__":
    main()
