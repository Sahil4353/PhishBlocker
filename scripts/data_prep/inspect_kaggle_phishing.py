from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

RAW_KAGGLE_DIR = Path("data/raw/kaggle_phishing")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


TEXT_CANDIDATES = [
    "email text",
    "email_text",
    "body",
    "text",
    "content",
    "message",
    "body_text",
]

LABEL_CANDIDATES = [
    "email type",
    "email_type",
    "label",
    "spam/ham",
    "class",
    "category",
    "target",
]


def find_candidate_cols(cols: List[str], candidates: List[str]) -> List[str]:
    found = []
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        for lc, orig in lower_map.items():
            if lc == cand.lower():
                found.append(orig)
    return found


def inspect_file(path: Path) -> None:
    logger = logging.getLogger("inspect_kaggle")
    logger.info("====== %s ======", path)

    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to read %s: %s", path, e)
        return

    rows, cols = df.shape
    logger.info("Shape: %d rows x %d columns", rows, cols)
    logger.info("Columns: %s", list(df.columns))

    text_cols = find_candidate_cols(list(df.columns), TEXT_CANDIDATES)
    label_cols = find_candidate_cols(list(df.columns), LABEL_CANDIDATES)

    logger.info("Possible text columns: %s", text_cols or "[]")
    logger.info("Possible label columns: %s", label_cols or "[]")

    # show value counts for possible label columns
    for col in label_cols:
        logger.info("Value counts for [%s]:", col)
        vc = df[col].value_counts(dropna=False)
        logger.info("\n%s", vc)

    # small sample
    logger.info("Sample rows:")
    with pd.option_context("display.max_colwidth", 120):
        logger.info("\n%s", df.head(3))


def main() -> None:
    setup_logging()
    logger = logging.getLogger("inspect_kaggle")

    logger.info("Inspecting Kaggle phishing CSVs under %s", RAW_KAGGLE_DIR.resolve())

    if not RAW_KAGGLE_DIR.exists():
        logger.error("RAW_KAGGLE_DIR does not exist.")
        return

    csvs = sorted(RAW_KAGGLE_DIR.glob("*.csv"))
    if not csvs:
        logger.error("No CSV files found in %s", RAW_KAGGLE_DIR)
        return

    for path in csvs:
        inspect_file(path)


if __name__ == "__main__":
    main()
