from __future__ import annotations

import logging
import mailbox
import os
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd

RAW_DIR = Path("data/raw/nazario")
OUT_PATH = Path("data/processed/clean_phishing.csv")


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("nazario_phishing")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    handler.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def extract_body_from_message(msg) -> str:
    """Extract a text body from an email.message.Message (mbox entry)."""
    parts: List[str] = []

    try:
        if msg.is_multipart():
            for part in msg.walk():
                ctype = (part.get_content_type() or "").lower()
                if not ctype.startswith("text/"):
                    continue
                if part.get_filename():
                    continue  # skip attachments

                payload = part.get_payload(decode=True)
                if payload is None:
                    continue

                charset = part.get_content_charset() or "utf-8"
                try:
                    parts.append(payload.decode(charset, errors="replace"))
                except Exception:
                    parts.append(payload.decode("utf-8", errors="replace"))
        else:
            payload = msg.get_payload(decode=True)
            if payload is not None:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    parts.append(payload.decode(charset, errors="replace"))
                except Exception:
                    parts.append(payload.decode("utf-8", errors="replace"))
            else:
                text = msg.get_payload()
                if isinstance(text, str):
                    parts.append(text)
    except Exception:
        try:
            parts.append(str(msg))
        except Exception:
            pass

    return "\n".join(parts).strip()


def iter_mbox_messages(mbox_path: Path, log: logging.Logger):
    """Yield (index, body_text) pairs from a single mbox-like file."""
    try:
        mbox = mailbox.mbox(mbox_path, create=False)
    except Exception as e:
        log.error("Failed to open %s as mbox: %s", mbox_path, e)
        return

    for idx, msg in enumerate(mbox):
        try:
            body = extract_body_from_message(msg)
        except Exception as e:
            log.warning(
                "Failed to extract message #%d in %s: %s",
                idx,
                mbox_path.name,
                e,
            )
            continue
        yield idx, body


def main() -> None:
    log = setup_logging()
    log.info("Python %s on os.name=%s", sys.version.split()[0], os.name)
    log.info("RAW_DIR = %s", RAW_DIR.resolve())
    log.info("OUT_PATH = %s", OUT_PATH.resolve())

    if not RAW_DIR.exists():
        log.error("RAW_DIR does not exist. Did you run fetch_nazario.ps1?")
        sys.exit(1)

    files = sorted(
        [
            p
            for p in RAW_DIR.iterdir()
            if p.is_file()
            and not p.name.lower().startswith("readme")
            and p.name not in {"phishing", "~jose_phishing"}
        ],
        key=lambda x: x.name,
    )

    if not files:
        log.error("No data files found in %s", RAW_DIR)
        sys.exit(1)

    log.info("Found %d Nazario corpus files:", len(files))
    for f in files:
        log.info("  - %s (%.2f MB)", f.name, f.stat().st_size / (1024 * 1024))

    rows: List[Dict[str, str]] = []
    total_msgs = 0
    total_nonempty = 0

    for fpath in files:
        log.info("Processing file: %s", fpath.name)
        for idx, body in iter_mbox_messages(fpath, log):
            total_msgs += 1
            if body:
                total_nonempty += 1

            rows.append(
                {
                    "body_text": body,
                    "label": "phishing",
                    "source_file": fpath.name,
                    "source_index": idx,
                }
            )

            if total_msgs % 10_000 == 0:
                log.info(
                    "Progress: total_msgs=%d, non_empty=%d (last file: %s)",
                    total_msgs,
                    total_nonempty,
                    fpath.name,
                )

    log.info(
        "Finished reading Nazario corpus: total_msgs=%d, non_empty=%d",
        total_msgs,
        total_nonempty,
    )

    if not rows:
        log.warning("No messages extracted. Writing empty CSV.")
        df = pd.DataFrame(columns=["body_text", "label", "source_file", "source_index"])
    else:
        df = pd.DataFrame(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    log.info("[nazario] wrote %s rows=%d (all labeled 'phishing')", OUT_PATH, len(df))

    with pd.option_context("display.max_colwidth", 200):
        log.info("Sample rows:\n%s", df.head(3))


if __name__ == "__main__":
    main()
