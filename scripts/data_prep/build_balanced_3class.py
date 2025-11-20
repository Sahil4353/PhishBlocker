from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROCESSED_DIR = Path("data/processed")

ENRON_CSV = PROCESSED_DIR / "clean_enron.csv"
SPAMASSASSIN_CSV = PROCESSED_DIR / "clean_spam.csv"
NAZARIO_CSV = PROCESSED_DIR / "clean_phishing.csv"
KAGGLE_CSV = PROCESSED_DIR / "clean_kaggle_phishing.csv"

OUT_BALANCED = PROCESSED_DIR / "clean_all_balanced_3class.csv"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_source(path: Path, source_name: str) -> pd.DataFrame:
    logger = logging.getLogger("build_balanced")

    if not path.exists():
        logger.error("Expected processed file not found: %s", path)
        sys.exit(1)

    df = pd.read_csv(path)
    cols = list(df.columns)
    if "body_text" not in cols or "label" not in cols:
        logger.error(
            "[%s] missing required columns 'body_text'/'label' in %s. Columns: %s",
            source_name,
            path,
            cols,
        )
        sys.exit(1)

    df = df[["body_text", "label"]].copy()
    df["__source_dataset"] = source_name
    logger.info("[%s] loaded %s rows", source_name, len(df))
    return df


def main() -> None:
    setup_logging()
    logger = logging.getLogger("build_balanced")

    logger.info("Loading processed datasets from %s", PROCESSED_DIR.resolve())

    parts: List[pd.DataFrame] = []
    parts.append(load_source(ENRON_CSV, "enron"))
    parts.append(load_source(SPAMASSASSIN_CSV, "spamassassin"))
    parts.append(load_source(NAZARIO_CSV, "nazario"))
    parts.append(load_source(KAGGLE_CSV, "kaggle_phishing"))

    combined = pd.concat(parts, ignore_index=True)
    logger.info("Combined rows before filtering labels: %d", len(combined))

    # Keep only our 3 labels
    valid_labels = {"safe", "spam", "phishing"}
    combined = combined[combined["label"].isin(valid_labels)].copy()

    logger.info(
        "After filtering to labels %s, rows=%d",
        valid_labels,
        len(combined),
    )

    logger.info("Label counts (combined):\n%s", combined["label"].value_counts())

    # Determine balanced size per class
    counts = combined["label"].value_counts().to_dict()
    if len(counts) < 3:
        logger.error("Expected 3 labels (safe/spam/phishing) but found: %s", counts)
        sys.exit(1)

    n_per_class = min(counts.values())
    logger.info("Balancing to n_per_class=%d (min over classes)", n_per_class)

    # Stratified sampling
    balanced_parts: List[pd.DataFrame] = []
    for label in sorted(valid_labels):
        df_label = combined[combined["label"] == label]
        if len(df_label) < n_per_class:
            logger.error(
                "Label %r has only %d rows (<%d). This should not happen.",
                label,
                len(df_label),
                n_per_class,
            )
            sys.exit(1)

        sampled = df_label.sample(n=n_per_class, random_state=42)
        balanced_parts.append(sampled)
        logger.info(
            "Sampled %d rows for label=%r (from %d available)",
            n_per_class,
            label,
            len(df_label),
        )

    balanced = pd.concat(balanced_parts, ignore_index=True)
    # Shuffle final
    balanced = balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

    logger.info("Final balanced dataset shape: %s", balanced.shape)
    logger.info("Final label counts:\n%s", balanced["label"].value_counts())

    OUT_BALANCED.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(OUT_BALANCED, index=False)
    logger.info("Wrote balanced dataset to %s", OUT_BALANCED)


if __name__ == "__main__":
    main()
