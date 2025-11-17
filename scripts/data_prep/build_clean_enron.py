# scripts/data_prep/build_clean_enron.py
from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator
import re  # needed for HEADER_CUTOFF_RE

import pandas as pd

RAW_DIR = Path("data/raw/enron/maildir")
OUT_PATH = Path("data/processed/clean_enron.csv")

TARGET_COUNT = 20_000
OVERSAMPLE_FACTOR = 2  # sample 40k, then trim to 20k
PROGRESS_EVERY = 1_000
FAILED_LOG_LIMIT = 20

HEADER_CUTOFF_RE = re.compile(r"\n\n", re.MULTILINE)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def extract_body(text: str) -> str:
    """Drop headers before the first blank line and keep the body."""
    parts = HEADER_CUTOFF_RE.split(text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]
    return body.strip()


def iter_all_files(base: Path) -> Iterator[Path]:
    """Yield every file under base."""
    for root, _, files in os.walk(base):
        rp = Path(root)
        for fn in files:
            yield rp / fn


def read_and_extract(path: Path, raw_root_verbatim: str | None, raw_dir_resolved: Path) -> tuple[Path, str]:
    """
    Read file text and extract body.

    On Windows, if raw_root_verbatim is provided, use a \\?\ path so filenames
    ending with '.' work properly.
    """
    if os.name == "nt" and raw_root_verbatim is not None:
        # Build verbatim path: \\?\C:\...\maildir\relative\file.
        rel = path.resolve().relative_to(raw_dir_resolved)
        raw_path = raw_root_verbatim + "\\" + str(rel)
        with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    else:
        raw = path.read_text(encoding="utf-8", errors="ignore")

    return path, extract_body(raw)


def main() -> None:
    setup_logging()
    logger = logging.getLogger("build_clean_enron")

    logger.info("Python %s on os.name=%s", sys.version.split()[0], os.name)
    logger.info("RAW_DIR = %s", RAW_DIR.resolve())
    logger.info("OUT_PATH = %s", OUT_PATH.resolve())

    if not RAW_DIR.exists():
        logger.error("RAW_DIR does not exist.")
        return

    raw_dir_resolved = RAW_DIR.resolve()
    raw_root_verbatim: str | None = None
    if os.name == "nt":
        # \\?\C:\Users\...\maildir
        raw_root_verbatim = r"\\?\{}".format(str(raw_dir_resolved))

    # 1) Collect all file paths
    logger.info("Scanning files…")
    t0 = time.time()
    all_paths = list(iter_all_files(RAW_DIR))
    total_files = len(all_paths)
    logger.info("Found %d files (%.2fs)", total_files, time.time() - t0)

    if total_files == 0:
        logger.error("No files found under RAW_DIR, aborting.")
        return

    # 2) Sample candidate files
    random.seed(42)
    num_candidates = min(total_files, TARGET_COUNT * OVERSAMPLE_FACTOR)
    logger.info("Sampling %d candidate files…", num_candidates)
    candidate_paths = random.sample(all_paths, num_candidates)

    # 3) Parallel read with ThreadPoolExecutor
    max_workers = min(32, (os.cpu_count() or 4) * 2)
    logger.info("ThreadPoolExecutor workers=%d", max_workers)

    rows: list[dict[str, str]] = []
    processed = 0
    non_empty = 0
    empty = 0
    failed_reads = 0
    failed_logs_shown = 0

    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(read_and_extract, p, raw_root_verbatim, raw_dir_resolved): p
            for p in candidate_paths
        }

        for fut in concurrent.futures.as_completed(futures):
            fpath = futures[fut]
            try:
                p, body = fut.result()
            except Exception as e:  # noqa: BLE001
                failed_reads += 1
                if failed_logs_shown < FAILED_LOG_LIMIT:
                    logger.warning("Failed read [%s]: %s", fpath, e)
                    failed_logs_shown += 1
                continue

            processed += 1
            body = body.strip()

            if body:
                non_empty += 1
                rows.append(
                    {
                        "body_text": body,
                        "label": "safe",
                        "source_file": str(p.relative_to(RAW_DIR)),
                    }
                )
            else:
                empty += 1

            if (
                processed <= 100
                or processed % PROGRESS_EVERY == 0
                or processed == num_candidates
            ):
                elapsed = time.time() - start
                rate = processed / max(1e-6, elapsed)
                logger.info(
                    "Progress %d/%d (%.2f%%) | non_empty=%d | empty=%d | failed=%d | rate=%.1f/s",
                    processed,
                    num_candidates,
                    processed / num_candidates * 100.0,
                    non_empty,
                    empty,
                    failed_reads,
                    rate,
                )

    logger.info(
        "Done reading: processed=%d | non_empty=%d | empty=%d | failed=%d",
        processed,
        non_empty,
        empty,
        failed_reads,
    )

    # 4) Shuffle + trim to TARGET_COUNT
    if len(rows) == 0:
        logger.error("No usable emails collected. Nothing to write.")
        print("[enron] collected=0 safe emails")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
        print(f"[enron] wrote {OUT_PATH}")
        return

    if len(rows) < TARGET_COUNT:
        logger.warning(
            "Only %d usable emails found (<%d). Keeping all.",
            len(rows),
            TARGET_COUNT,
        )
        final_rows = rows
    else:
        random.shuffle(rows)
        final_rows = rows[:TARGET_COUNT]
        logger.info("Trimmed to %d safe emails.", TARGET_COUNT)

    # 5) Write CSV
    df = pd.DataFrame(final_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"[enron] collected={len(final_rows)} safe emails")
    print(f"[enron] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
# EOF