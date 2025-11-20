from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

RAW_190K_DIR = Path("data/raw/kaggle_190k")
INPUT_FILE = RAW_190K_DIR / "spam_Emails_data.csv"  # matches your filename

OUT_PATH = Path("data/processed/clean_kaggle_190k_spam.csv")


def setup_logging():
    logger = logging.getLogger("build_clean_kaggle_190k_spam")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


def main():
    log = setup_logging()
    log.info("Loading dataset from %s", INPUT_FILE.resolve())

    if not INPUT_FILE.exists():
        log.error("Input file does not exist: %s", INPUT_FILE)
        sys.exit(1)

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        log.exception("Failed to load CSV: %s", e)
        sys.exit(1)

    log.info("Loaded CSV: shape=%s", df.shape)
    log.info("Columns: %s", list(df.columns))

    # Verify columns
    if "label" not in df.columns or "text" not in df.columns:
        log.error("Expected columns 'label' and 'text' not found.")
        sys.exit(1)

    # Filter only spam rows
    spam_df = df[df["label"].str.strip().str.lower() == "spam"].copy()
    log.info("Filtered spam rows: %d", len(spam_df))

    # Clean text
    spam_df["body_text"] = spam_df["text"].astype(str).fillna("").str.strip()
    spam_df["label"] = "spam"
    spam_df["source_path"] = INPUT_FILE.name

    final = spam_df[["body_text", "label", "source_path"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_PATH, index=False)

    log.info(
        "[kaggle_190k_spam] wrote %s rows=%d",
        OUT_PATH,
        len(final),
    )


if __name__ == "__main__":
    main()
