from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROCESSED_DIR = Path("data/processed")

FILES: Dict[str, Path] = {
    "enron": PROCESSED_DIR / "clean_enron.csv",
    "spamassassin": PROCESSED_DIR / "clean_spam.csv",
    "nazario": PROCESSED_DIR / "clean_phishing.csv",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def truncate(text: str, max_len: int = 120) -> str:
    if not isinstance(text, str):
        return str(text)
    if len(text) <= max_len:
        return text.replace("\n", " ")
    return text[: max_len - 3].replace("\n", " ") + "..."


def inspect_single(name: str, path: Path) -> pd.DataFrame | None:
    logger = logging.getLogger("inspect")
    logger.info("====== [%s] %s ======", name.upper(), path)

    if not path.exists():
        logger.error("File does NOT exist: %s", path)
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to read %s: %s", path, e)
        return None

    rows, cols = df.shape
    logger.info("Shape: %d rows x %d columns", rows, cols)
    logger.info("Columns: %s", list(df.columns))

    # Basic checks for required columns
    missing_cols: List[str] = []
    for col in ["body_text", "label"]:
        if col not in df.columns:
            missing_cols.append(col)
    if missing_cols:
        logger.warning(
            "[%s] Missing expected columns: %s", name, ", ".join(missing_cols)
        )

    # Label distribution
    if "label" in df.columns:
        logger.info("Label value counts:")
        label_counts = df["label"].value_counts(dropna=False)
        for label, cnt in label_counts.items():
            logger.info("  %r: %d", label, cnt)

    # Body length stats
    if "body_text" in df.columns:
        lengths = df["body_text"].fillna("").astype(str).str.len()
        desc = lengths.describe()
        logger.info(
            "body_text length stats: count=%d min=%d max=%d mean=%.1f",
            int(desc["count"]),
            int(desc["min"]),
            int(desc["max"]),
            float(desc["mean"]),
        )

        num_empty = (lengths == 0).sum()
        num_na = df["body_text"].isna().sum()
        logger.info(
            "Empty body_text rows: %d | NaN body_text rows: %d", num_empty, num_na
        )

    # Show a small sample
    logger.info("Sample rows (truncated body_text):")
    sample = df.head(5).copy()
    if "body_text" in sample.columns:
        sample["body_text"] = sample["body_text"].fillna("").astype(str).apply(truncate)
    logger.info("\n%s", sample)

    return df


def main() -> None:
    setup_logging()
    logger = logging.getLogger("inspect")

    logger.info("Inspecting processed datasets under %s", PROCESSED_DIR.resolve())

    all_dfs: List[pd.DataFrame] = []

    for name, path in FILES.items():
        df = inspect_single(name, path)
        if df is not None:
            # Add a column so we know the origin when concatenated
            df = df.copy()
            df["__source_dataset"] = name
            all_dfs.append(df)

    if not all_dfs:
        logger.error("No datasets could be loaded. Nothing to summarize.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info("====== OVERALL SUMMARY ======")
    logger.info("Total combined rows from all datasets: %d", len(combined))

    if "label" in combined.columns:
        logger.info("Overall label counts across all datasets:")
        label_counts = combined["label"].value_counts(dropna=False)
        for label, cnt in label_counts.items():
            logger.info("  %r: %d", label, cnt)

        # Label counts per dataset
        logger.info("Label counts per dataset:")
        per_ds = (
            combined.groupby(["__source_dataset", "label"])
            .size()
            .reset_index(name="count")
        )
        logger.info("\n%s", per_ds)

    # Check for unexpected labels
    expected_labels = {"safe", "spam", "phishing"}
    if "label" in combined.columns:
        unique_labels = set(combined["label"].dropna().unique().tolist())
        unexpected = unique_labels - expected_labels
        if unexpected:
            logger.warning(
                "Unexpected labels found: %s", ", ".join(sorted(map(str, unexpected)))
            )
        else:
            logger.info(
                "All labels are in the expected set: %s",
                ", ".join(sorted(expected_labels)),
            )


if __name__ == "__main__":
    main()
