# scripts/data_prep/build_clean_spam.py
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd

# SpamAssassin raw location (matches your PowerShell script)
RAW_DIR = Path("data/raw/spamassassin")
OUT_PATH = Path("data/processed/clean_spam.csv")

# Headers end at first blank line (handle \r\n or \n)
HEADER_CUTOFF_RE = re.compile(r"\r?\n\r?\n", re.MULTILINE)

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger for CLI usage.
    If logging is already configured, just bump the level.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def extract_body(text: str) -> str:
    """
    Drop headers before the first blank line and keep the body.
    """
    parts = HEADER_CUTOFF_RE.split(text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]
    body = body.strip()

    if not body:
        logger.debug("Extracted empty body from message")
    return body


def infer_label_from_path(path: Path) -> str | None:
    """
    Map SpamAssassin folder names → labels for this dataset:

        spam, spam_2, etc.     → 'spam'
        easy_ham, hard_ham, …  → 'safe'

    We only look at the *immediate parent directory* to avoid matching
    the top-level 'spamassassin' folder name.
    """
    parent = path.name.lower()  # e.g. 'easy_ham', 'spam', 'hard_ham'

    if parent.startswith("spam"):
        return "spam"

    if "ham" in parent:
        return "safe"

    # Unknown folder: we can either default to safe or skip.
    logger.warning(
        "Could not infer label from folder %s (full path=%s); skipping files under it.",
        parent,
        path,
    )
    return None


def main() -> None:
    setup_logging()  # set to DEBUG if you want more detail

    logger.info("Starting build_clean_spam from RAW_DIR=%s", RAW_DIR)

    if not RAW_DIR.exists():
        logger.error("Raw directory does not exist: %s", RAW_DIR)
        raise SystemExit(1)

    rows: list[dict[str, str]] = []

    total_files = 0
    processed_files = 0
    unreadable_files = 0
    empty_bodies = 0
    skipped_unknown_folder = 0
    skipped_hidden = 0

    for root, _, files in os.walk(RAW_DIR):
        root_path = Path(root)

        # Determine label for this folder
        label = infer_label_from_path(root_path.relative_to(RAW_DIR))
        if label is None:
            skipped_unknown_folder += len(files)
            continue

        for fn in files:
            total_files += 1
            fpath = root_path / fn

            # Skip hidden/junk files such as .gitkeep
            if fn.startswith("."):
                skipped_hidden += 1
                continue

            # Skip obvious archive files if they exist alongside extracted mail
            if fpath.suffix in {".tar", ".bz2", ".gz", ".zip"}:
                logger.debug("Skipping archive file: %s", fpath)
                continue

            try:
                raw = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:  # noqa: BLE001
                unreadable_files += 1
                logger.warning("Skipping unreadable file %s: %s", fpath, exc)
                continue

            body = extract_body(raw)

            if not body:
                empty_bodies += 1
                # Optionally skip empty emails instead of keeping them
                continue

            rows.append(
                {
                    "body_text": body,
                    "label": label,  # 'safe' or 'spam'
                    "source_path": str(fpath.relative_to(RAW_DIR)),
                }
            )
            processed_files += 1

    if not rows:
        logger.error(
            "No rows generated from %s (total_files=%d, unreadable=%d)",
            RAW_DIR,
            total_files,
            unreadable_files,
        )
        raise SystemExit(1)

    logger.info(
        "Finished reading files: total=%d, processed=%d, "
        "unreadable=%d, empty_bodies_skipped=%d, "
        "skipped_hidden=%d, skipped_unknown_folder=%d",
        total_files,
        processed_files,
        unreadable_files,
        empty_bodies,
        skipped_hidden,
        skipped_unknown_folder,
    )

    try:
        df = pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to build DataFrame from collected rows: %s", exc)
        raise SystemExit(1)

    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_PATH, index=False)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to write CSV to %s: %s", OUT_PATH, exc)
        raise SystemExit(1)

    logger.info(
        "[spamassassin] wrote %s rows=%d | label_counts=%s",
        OUT_PATH,
        len(df),
        df["label"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()