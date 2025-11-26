from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROCESSED_DIR = Path("data/processed")

# Target rows per class (upper bound)
TARGET_PER_CLASS = 30_000

# Name for the large balanced dataset
OUT_BALANCED = PROCESSED_DIR / "clean_all_balanced_3class_30k.csv"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_source(path: Path) -> pd.DataFrame | None:
    logger = logging.getLogger("build_balanced")

    logger.info("Inspecting processed file: %s", path.name)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.warning("Failed to read %s: %s (skipping)", path, e)
        return None

    cols = list(df.columns)
    if "body_text" not in cols or "label" not in cols:
        logger.info(
            "Skipping %s: missing required 'body_text'/'label' columns. Columns: %s",
            path.name,
            cols,
        )
        return None

    df = df[["body_text", "label"]].copy()
    df["__source_dataset"] = path.stem  # e.g. 'clean_enron', 'clean_kaggle_190k_spam'
    logger.info("[%s] loaded %s rows", path.name, len(df))
    return df


def main() -> None:
    setup_logging()
    logger = logging.getLogger("build_balanced")

    logger.info("Loading processed datasets from %s", PROCESSED_DIR.resolve())

    if not PROCESSED_DIR.exists():
        logger.error("Processed directory does not exist: %s", PROCESSED_DIR)
        sys.exit(1)

    parts: List[pd.DataFrame] = []

    # Find all CSVs in processed/, skip previous balanced outputs
    csv_files = sorted(PROCESSED_DIR.glob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found in %s", PROCESSED_DIR)
        sys.exit(1)

    for path in csv_files:
        # Skip previously balanced datasets to avoid double-counting
        if path.name.startswith("clean_all_balanced"):
            logger.info("Skipping previously balanced file: %s", path.name)
            continue

        df = load_source(path)
        if df is not None:
            parts.append(df)

    if not parts:
        logger.error("No valid datasets loaded (none had body_text + label).")
        sys.exit(1)

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

    label_counts = combined["label"].value_counts()
    logger.info("Label counts (combined):\n%s", label_counts)

    # Determine balanced size per class
    counts = label_counts.to_dict()
    if len(counts) < 3:
        logger.error("Expected 3 labels (safe/spam/phishing) but found: %s", counts)
        sys.exit(1)

    min_available = min(counts.values())
    n_per_class = min(TARGET_PER_CLASS, min_available)
    logger.info(
        "Balancing to n_per_class=%d (TARGET_PER_CLASS=%d, min_available=%d)",
        n_per_class,
        TARGET_PER_CLASS,
        min_available,
    )

    # Stratified sampling
    balanced_parts: List[pd.DataFrame] = []
    for label in sorted(valid_labels):
        df_label = combined[combined["label"] == label]
        if len(df_label) < n_per_class:
            logger.error(
                "Label %r has only %d rows (<%d). This should not happen "
                "given min_available=%d.",
                label,
                len(df_label),
                n_per_class,
                min_available,
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
