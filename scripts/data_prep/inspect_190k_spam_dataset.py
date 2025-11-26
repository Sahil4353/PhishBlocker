from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# UPDATE THIS IF YOUR FILE HAS A DIFFERENT NAME
RAW_DIR = Path("data/raw/kaggle_190k")
INPUT_FILE = RAW_DIR / "spam_Emails_data.csv"   # <-- change this to your real file name


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    setup_logging()
    log = logging.getLogger("inspect_190k")

    log.info("Looking for file: %s", INPUT_FILE.resolve())

    if not INPUT_FILE.exists():
        log.error("File does not exist. Please update INPUT_FILE in this script.")
        return

    # Try loading CSV
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        log.exception("Failed to load CSV: %s", e)
        return

    log.info("Loaded CSV: shape=%s", df.shape)
    log.info("Columns: %s", list(df.columns))

    # Try detecting label & text columns
    TEXT_CANDIDATES = ["text", "body", "message", "email", "content"]
    LABEL_CANDIDATES = ["label", "spam", "category", "is_spam"]

    text_col = None
    label_col = None

    lower_cols = {c.lower(): c for c in df.columns}

    for cand in TEXT_CANDIDATES:
        if cand in lower_cols:
            text_col = lower_cols[cand]
            break

    for cand in LABEL_CANDIDATES:
        if cand in lower_cols:
            label_col = lower_cols[cand]
            break

    log.info("Detected text column: %s", text_col)
    log.info("Detected label column: %s", label_col)

    if label_col is not None:
        log.info("Label value counts:\n%s", df[label_col].value_counts(dropna=False))

    # Show a small sample
    with pd.option_context("display.max_colwidth", 150):
        log.info("Sample rows:\n%s", df.head(5))


if __name__ == "__main__":
    main()
