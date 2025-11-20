from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

RAW_KAGGLE_DIR = Path("data/raw/kaggle_phishing")
KAGGLE_MAIN_FILE = RAW_KAGGLE_DIR / "phishing_email.csv"

OUT_PATH = Path("data/processed/clean_kaggle_phishing.csv")


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("build_clean_kaggle_phishing")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    handler.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def map_kaggle_label(raw_val) -> str | None:
    """
    Map Kaggle 0/1 label -> our labels.

    In this dataset:
      0 = legitimate/safe
      1 = phishing/spam

    We will treat:
      0 -> 'safe'
      1 -> 'phishing'
    """
    if pd.isna(raw_val):
        return None

    try:
        v = int(raw_val)
    except (TypeError, ValueError):
        v = None

    if v == 0:
        return "safe"
    if v == 1:
        return "phishing"

    return None


def main() -> None:
    log = setup_logging()
    log.info("RAW_KAGGLE_DIR = %s", RAW_KAGGLE_DIR.resolve())
    log.info("KAGGLE_MAIN_FILE = %s", KAGGLE_MAIN_FILE)

    if not KAGGLE_MAIN_FILE.exists():
        log.error("Main Kaggle file does not exist: %s", KAGGLE_MAIN_FILE)
        sys.exit(1)

    try:
        df = pd.read_csv(KAGGLE_MAIN_FILE)
    except Exception as e:  # noqa: BLE001
        log.exception("Failed to read %s: %s", KAGGLE_MAIN_FILE, e)
        sys.exit(1)

    log.info("Loaded Kaggle file: shape=%s", df.shape)
    log.info("Columns: %s", list(df.columns))

    # Hard-check expected columns
    if "text_combined" not in df.columns or "label" not in df.columns:
        log.error(
            "Expected columns 'text_combined' and 'label' not found. "
            "Columns present: %s",
            list(df.columns),
        )
        sys.exit(1)

    # Map labels
    df["label_mapped"] = df["label"].apply(map_kaggle_label)
    before = len(df)
    df = df[df["label_mapped"].notna()].copy()
    after = len(df)

    log.info("Rows before label mapping: %d, after filtering: %d", before, after)
    log.info("Mapped label counts:\n%s", df["label_mapped"].value_counts())

    # Build unified schema
    df["body_text"] = df["text_combined"].astype(str).fillna("").str.strip()
    df["label"] = df["label_mapped"]
    df["source_path"] = KAGGLE_MAIN_FILE.name

    final = df[["body_text", "label", "source_path"]].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_PATH, index=False)

    log.info(
        "[kaggle_phishing] wrote %s rows=%d | label_counts=%s",
        OUT_PATH,
        len(final),
        final["label"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()
