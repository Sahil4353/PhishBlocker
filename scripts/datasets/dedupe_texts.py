#!/usr/bin/env python
"""
Exact dedupe by normalized text (case-fold + whitespace collapse), per label.
Keeps the first occurrence of (text_hash, label). Emits summary stats.

Usage:
  python scripts/datasets/dedupe_texts.py ^
    --inputs data/processed/spamassassin.csv data/processed/nazario.csv data/processed/enron.csv ^
    --out    data/processed/mix_dedup.csv
"""

from __future__ import annotations
import argparse
import hashlib
import re
from pathlib import Path
from typing import List

import pandas as pd


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.lower()
    s = s.replace("\r\n", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Exact dedupe by normalized text hash")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input CSVs with columns: body_text,label[,source,relpath,len_chars]",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path")
    args = ap.parse_args()

    # Load all inputs
    dfs: List[pd.DataFrame] = []
    for inp in args.inputs:
        df = pd.read_csv(inp)
        missing = [c for c in ("body_text", "label") if c not in df.columns]
        if missing:
            raise SystemExit(f"{inp} is missing required columns: {missing}")
        df["__source_csv"] = inp
        dfs.append(df)

    if not dfs:
        raise SystemExit("No inputs provided")

    all_df = pd.concat(dfs, ignore_index=True)
    total_in = len(all_df)

    # Normalize + hash
    all_df["__norm"] = all_df["body_text"].astype(str).map(normalize_text)
    all_df["__hash"] = all_df["__norm"].map(sha256_hex)

    # Drop exact duplicates per (hash, label) keeping first occurrence
    before = len(all_df)
    all_df = all_df.drop_duplicates(subset=["__hash", "label"], keep="first")
    after = len(all_df)
    removed = before - after

    # Optional: keep a quick length for downstream sanity (unchanged if present)
    if "len_chars" not in all_df.columns:
        all_df["len_chars"] = all_df["__norm"].str.len()

    # Tidy
    out_cols = [
        c
        for c in [
            "body_text",
            "label",
            "source",
            "relpath",
            "len_chars",
            "__source_csv",
        ]
        if c in all_df.columns
    ]
    out_df = all_df[out_cols].copy()

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    # Report
    by_label_in = pd.Series(dfs[0].columns)  # dummy to silence pyright; not used
    print(f"Total in:     {total_in}")
    print(f"Total unique: {after}")
    print(f"Removed dups: {removed}")
    print("By label (post-dedupe):")
    print(out_df["label"].value_counts().sort_index())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
