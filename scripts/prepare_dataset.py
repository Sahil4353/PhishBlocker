#!/usr/bin/env python
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# --- helpers ---------------------------------------------------------------

URL_RE = re.compile(r"https?://[^\s)>\]\"']+", re.IGNORECASE)
SHORTENERS = {
    "bit.ly",
    "t.co",
    "goo.gl",
    "tinyurl.com",
    "is.gd",
    "ow.ly",
    "rebrand.ly",
    "buff.ly",
}

REQUIRED_COLS = [
    "id",
    "source",
    "label",
    "subject",
    "sender",
    "recipients",
    "body_text",
    "html_raw",
    "timestamp",
]


def html_to_text(html: str) -> str:
    if not isinstance(html, str) or not html.strip():
        return ""
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.extract()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def extract_urls(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return [m.group(0).strip(").,>]}\"'") for m in URL_RE.finditer(text)]


def normalize_sender(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    # Very light normalization; RFC parsing can come later
    return s


def content_hash(subject: str, body_text: str) -> str:
    h = hashlib.sha256()
    h.update((subject or "").encode("utf-8"))
    h.update(b"\n")
    h.update((body_text or "").encode("utf-8"))
    return h.hexdigest()[:16]


def safe_listify(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.strip():
        # split recipients on comma/semicolon
        return [p.strip() for p in re.split(r"[;,]", x) if p.strip()]
    return []


def has_shortener(url: str) -> bool:
    try:
        host = re.sub(r"^https?://", "", url).split("/")[0].lower()
    except Exception:
        return False
    return host in SHORTENERS


# --- main transform --------------------------------------------------------


def load_and_normalize(input_path: Path) -> pd.DataFrame:
    # Accept CSV or JSONL with flexible incoming columns; map to REQUIRED_COLS
    if input_path.suffix.lower() in {".json", ".jsonl"}:
        df = pd.read_json(input_path, lines=True)
    else:
        df = pd.read_csv(input_path)

    # Best-effort column mapping
    colmap = {
        "from": "sender",
        "to": "recipients",
        "body": "body_text",
        "html": "html_raw",
        "date": "timestamp",
        "time": "timestamp",
    }
    for src, dst in colmap.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # Ensure required columns exist
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = ""

    # Derive body_text from html_raw if empty
    missing_text = df["body_text"].isna() | (df["body_text"].str.len() == 0)
    df.loc[missing_text, "body_text"] = df.loc[missing_text, "html_raw"].apply(
        html_to_text
    )

    # Normalize sender/recipients
    df["sender"] = df["sender"].map(normalize_sender)
    df["recipients"] = df["recipients"].map(safe_listify)

    # URL extraction (from text + html_text)
    df["urls"] = df["body_text"].map(extract_urls)
    # simple url features (we'll extend later)
    df["url_count"] = df["urls"].map(len)
    df["has_shortener"] = df["urls"].map(lambda us: any(has_shortener(u) for u in us))

    # Lightweight dedup on (subject, body_text)
    df["dup_key"] = [content_hash(s, b) for s, b in zip(df["subject"], df["body_text"])]
    before = len(df)
    df = df.drop_duplicates(subset=["dup_key"]).reset_index(drop=True)
    after = len(df)
    print(f"De-duplicated: {before - after} removed; {after} remain.")

    # Basic label cleanup
    df["label"] = (
        df["label"]
        .str.strip()
        .str.lower()
        .replace({"junk": "spam", "spam/phish": "phish", "malicious": "phish"})
    )

    return df


def stratified_split(df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42):
    # Keep only rows that have one of the supported labels
    df = df[df["label"].isin(["ham", "spam", "phish"])].copy()
    # Simple stratified split
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df, test_size=val_fraction, random_state=seed, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# --- CLI -------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Prepare email dataset for ML.")
    ap.add_argument(
        "--input",
        required=True,
        help="Path to raw CSV/JSONL (e.g., data/raw/kaggle.csv)",
    )
    ap.add_argument("--outdir", default="data/processed", help="Output directory")
    ap.add_argument(
        "--val", type=float, default=0.2, help="Validation fraction (default 0.2)"
    )
    args = ap.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_normalize(inp)
    train_df, val_df = stratified_split(df, args.val)

    train_path = outdir / "train.csv"
    val_path = outdir / "val.csv"
    schema_path = outdir / "schema.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    schema = {
        "required_columns": REQUIRED_COLS,
        "derived_columns": ["urls", "url_count", "has_shortener", "dup_key"],
        "label_values": ["ham", "spam", "phish"],
        "counts": {
            "train": len(train_df),
            "val": len(val_df),
        },
    }
    schema_path.write_text(json.dumps(schema, indent=2))
    print(f"Wrote {train_path} ({len(train_df)} rows)")
    print(f"Wrote {val_path} ({len(val_df)} rows)")
    print(f"Wrote {schema_path}")


if __name__ == "__main__":
    sys.exit(main())
