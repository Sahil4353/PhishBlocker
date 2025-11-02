#!/usr/bin/env python
from __future__ import annotations
import os
import argparse
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Iterable, Tuple, List, Union, Optional, cast

import email
import pandas as pd
from email import policy
from email.message import EmailMessage

# ----------------------------
# Logging setup
# ----------------------------

logger = logging.getLogger("normalize_eml")
_PROJECT_PARSER_LOGGED = False


def setup_logging(level: str, debug_flag: bool) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if debug_flag:
        logger.setLevel(logging.DEBUG)


# ----------------------------
# Bytes/str safe extractors
# ----------------------------

BytesLike = Union[bytes, bytearray, memoryview]


def _strip_html(html: str) -> str:
    # Minimal HTML → text (no external deps)
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = unescape(html)
    html = re.sub(r"\s+", " ", html).strip()
    return html


def safe_relpath(p: Path, base: Path) -> str:
    """Return p relative to base, tolerating absolute/relative mix and Windows quirks."""
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except Exception:
        try:
            return os.path.relpath(str(p), start=str(base))
        except Exception:
            # last resort: fall back to filename
            return p.name


def _best_text_from_message(msg: EmailMessage) -> str:
    """
    Prefer text/plain; else use text/html (stripped).
    If single-part and type startswith text/, accept it.
    As a last resort, decode payload bytes.
    """
    try:
        if msg.is_multipart():
            plain_parts: List[str] = []
            html_parts: List[str] = []
            for part in msg.walk():
                p = cast(EmailMessage, part)
                ctype = (p.get_content_type() or "").lower()
                if ctype == "text/plain":
                    plain_parts.append(p.get_content() or "")
                elif ctype == "text/html":
                    html_parts.append(p.get_content() or "")
            if plain_parts:
                return "\n".join(p.strip() for p in plain_parts if p).strip()
            if html_parts:
                html = "\n".join(h for h in html_parts if h)
                return _strip_html(html)
            # fallback: any text/* part
            any_text: List[str] = []
            for part in msg.walk():
                p = cast(EmailMessage, part)
                ctype = (p.get_content_type() or "").lower()
                if ctype.startswith("text/"):
                    any_text.append(p.get_content() or "")
            if any_text:
                return "\n".join(t.strip() for t in any_text if t).strip()
            return ""
        else:
            ctype = (msg.get_content_type() or "").lower()
            if ctype == "text/plain":
                return (msg.get_content() or "").strip()
            if ctype == "text/html":
                return _strip_html(msg.get_content() or "")
            if ctype.startswith("text/"):
                return (msg.get_content() or "").strip()
            # last resort: raw payload decode
            payload = msg.get_payload(decode=True)
            if isinstance(payload, (bytes, bytearray)):
                try:
                    return payload.decode("utf-8", "ignore").strip()
                except Exception:
                    return ""
            return ""
    except Exception as e:
        logger.debug("best_text_from_message failed: %s", e)
        return ""


def _extract_text_default(raw: Union[BytesLike, str]) -> str:
    """
    Safe parser that works for bytes/bytearray/memoryview/str.
    """
    try:
        if isinstance(raw, (bytes, bytearray, memoryview)):
            msg = cast(
                EmailMessage, email.message_from_bytes(raw, policy=policy.default)
            )
        else:
            msg = cast(
                EmailMessage, email.message_from_string(raw, policy=policy.default)
            )
        return _best_text_from_message(msg)
    except Exception as e:
        logger.debug("extract_text_default parse failure: %s", e)
        return ""


def _extract_text_with_app(raw: Union[BytesLike, str]) -> str:
    """
    Try project's parser first; fall back to default.
    """
    try:
        from app.services.parser import extract_body_text  # type: ignore

        txt = extract_body_text(raw)  # may accept bytes or str
        if not isinstance(txt, str):
            txt = str(txt or "")
        txt = txt.strip()
        if txt:
            return txt
        # If project parser returns empty, fall back:
        return _extract_text_default(raw)
    except Exception as e:
        global _PROJECT_PARSER_LOGGED
        if not _PROJECT_PARSER_LOGGED:
            logger.debug("project parser unavailable or failed: %s", e)
            _PROJECT_PARSER_LOGGED = True
        return _extract_text_default(raw)


# ----------------------------
# File iteration helpers
# ----------------------------

ARCHIVE_SUFFIXES = (".tar", ".gz", ".bz2", ".xz", ".zip", ".7z")


def is_archive(p: Path) -> bool:
    name = p.name.lower()
    if name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        return True
    return p.suffix.lower() in ARCHIVE_SUFFIXES


def iter_message_files(root: Path) -> Iterable[Path]:
    """
    Windows-safe file discovery:
    - On Windows, walk with \\?\\ extended-length paths so trailing-dot files are yielded.
    - On other platforms, fall back to pathlib.rglob.
    - Skip archives; otherwise yield all files (filtering happens in parsing).
    """
    if sys.platform.startswith("win"):
        # Build extended-length absolute root (\\?\C:\...\maildir)
        abs_root = str(root.resolve())
        if not abs_root.startswith("\\\\?\\"):
            walk_root = "\\\\?\\" + abs_root
        else:
            walk_root = abs_root

        for dirpath, dirnames, filenames in os.walk(walk_root):
            # Convert back to normal form for Path/relpath ops
            normal_dirpath = dirpath[4:] if dirpath.startswith("\\\\?\\") else dirpath
            for fn in filenames:
                normal_full = os.path.join(normal_dirpath, fn)
                p = Path(normal_full)
                if is_archive(p):
                    continue
                yield p
        return

    # Non-Windows: pathlib is fine
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if is_archive(p):
            continue
        yield p


# ----------------------------
# Diagnostics & counters
# ----------------------------


@dataclass
class Counters:
    files_seen: int = 0
    archives_skipped: int = 0
    candidates: int = 0
    extracted_ok: int = 0
    empty_after_parse: int = 0
    too_short: int = 0
    exceptions: int = 0
    ctype_counter: Counter = None  # type: ignore


def sniff_content_type(p: Path) -> Optional[str]:
    """
    Cheap, Windows-safe sniff of Content-Type from the header only.
    - Reads only the first ~8 KiB (not the whole file).
    - Uses \\?\\ extended path on Windows so trailing-dot files work.
    - Falls back to None on any error quickly.
    """
    try:
        if sys.platform.startswith("win"):
            abs_str = str(p.resolve())
            if not abs_str.startswith("\\\\?\\"):
                win_path = "\\\\?\\" + abs_str
            else:
                win_path = abs_str
            with open(win_path, "rb") as f:
                head = f.read(8192)
        else:
            with open(p, "rb") as f:
                head = f.read(8192)

        # Split headers from body if possible
        if b"\r\n\r\n" in head:
            headers = head.split(b"\r\n\r\n", 1)[0]
        elif b"\n\n" in head:
            headers = head.split(b"\n\n", 1)[0]
        else:
            headers = head

        # Handle folded headers (continuation lines)
        lines = []
        for line in headers.splitlines():
            if lines and (line.startswith(b" ") or line.startswith(b"\t")):
                lines[-1] += b" " + line.strip()
            else:
                lines.append(line.strip())

        for line in lines:
            if line.lower().startswith(b"content-type:"):
                val = line.split(b":", 1)[1].strip()
                # Take token before ';' (ignore params)
                val = val.split(b";", 1)[0].strip()
                try:
                    return val.decode("utf-8", "ignore").lower()
                except Exception:
                    return None
        return None
    except Exception:
        return None


# ----------------------------
# Core normalization
# ----------------------------


def load_text(p: Path) -> str:
    r"""
    Read file robustly on Windows (supports trailing-dot names) and elsewhere.
    Uses \\?\ prefix on Windows to avoid path normalization that strips trailing dots.
    """
    try:
        if sys.platform.startswith("win"):
            # Build extended-length path: \\?\C:\abs\path\to\file.
            abs_str = str(p.resolve())
            if not abs_str.startswith("\\\\?\\"):
                win_path = "\\\\?\\" + abs_str
            else:
                win_path = abs_str
            with open(win_path, "rb") as f:
                raw = f.read()
        else:
            raw = p.read_bytes()
    except Exception as e:
        logger.debug("read failure: %s :: %s", p, e)
        return ""
    return _extract_text_with_app(raw)


def build_rows(
    paths: Iterable[Path],
    label: str,
    source: str,
    base: Path,
    min_chars: int,
    debug_sample: int = 0,
    ctype_sample: int = 0,
) -> Tuple[List[dict], Counters]:
    rows: List[dict] = []
    ctr = Counters(ctype_counter=Counter())
    sampled: List[Path] = []

    for p in paths:
        ctr.files_seen += 1

        # archive check (should already be filtered)
        if is_archive(p):
            ctr.archives_skipped += 1
            continue

        ctr.candidates += 1

        # optional content-type telemetry
        if ctype_sample and ctr.candidates <= ctype_sample:
            ctype = sniff_content_type(p)
            if ctype:
                ctr.ctype_counter[ctype] += 1

        try:
            text = load_text(p)
        except Exception as e:
            ctr.exceptions += 1
            logger.debug("extract exception: %s :: %s", p, e)
            continue

        if not text:
            ctr.empty_after_parse += 1
            continue

        text = text.replace("\r\n", "\n").strip()

        if len(text) < min_chars:
            ctr.too_short += 1
            continue

        if debug_sample and len(sampled) < debug_sample:
            sampled.append(p)

        rows.append(
            {
                "body_text": text,
                "label": label,
                "source": source,
                "relpath": safe_relpath(p, base),
                "len_chars": len(text),
            }
        )
        ctr.extracted_ok += 1

    # Print a compact source summary here; caller also prints totals.
    logger.info(
        "[%s] files_seen=%s candidates=%s ok=%s empty=%s short=%s exceptions=%s",
        source,
        ctr.files_seen,
        ctr.candidates,
        ctr.extracted_ok,
        ctr.empty_after_parse,
        ctr.too_short,
        ctr.exceptions,
    )

    if sampled:
        logger.debug("[%s] sample files:", source)
        for sp in sampled:
            logger.debug("  %s", sp)

    if ctr.ctype_counter:
        logger.debug(
            "[%s] content types (first %d candidates): %s",
            source,
            ctype_sample,
            dict(ctr.ctype_counter.most_common(10)),
        )

    return rows, ctr


def spamassassin_sets(sa_root: Path) -> List[Tuple[Iterable[Path], str]]:
    """
    Map SpamAssassin subfolders to labels.
    We allow extensionless files now via iter_message_files().
    """
    mapping: List[Tuple[Iterable[Path], str]] = []
    for name in ["easy_ham", "easy_ham_2", "hard_ham"]:
        d = sa_root / name
        if d.exists():
            mapping.append((iter_message_files(d), "safe"))
    for name in ["spam", "spam_2"]:
        d = sa_root / name
        if d.exists():
            mapping.append((iter_message_files(d), "spam"))
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize EML/maildir -> canonical CSVs")
    ap.add_argument("--spamassassin-root", type=Path, required=True)
    ap.add_argument("--nazario-root", type=Path, required=True)
    ap.add_argument("--enron-maildir", type=Path, required=True)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument(
        "--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR"
    )
    ap.add_argument(
        "--debug", action="store_true", help="Enable DEBUG logging and extra sampling"
    )
    ap.add_argument(
        "--debug-sample",
        type=int,
        default=5,
        help="Log up to N sample file paths per source when --debug",
    )
    ap.add_argument(
        "--ctype-sample",
        type=int,
        default=50,
        help="Sample size for content-type telemetry per source (0=off)",
    )
    args = ap.parse_args()

    setup_logging(args.log_level, args.debug)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Normalization start ===")
    logger.info(
        "roots: spamassassin=%s nazario=%s enron=%s",
        args.spamassassin_root,
        args.nazario_root,
        args.enron_maildir,
    )
    logger.info("min_chars=%d  out=%s", args.min_chars, out_dir)

    # SpamAssassin
    sa_rows: List[dict] = []
    sa_ctr_total = Counters(ctype_counter=Counter())
    for paths, lbl in spamassassin_sets(args.spamassassin_root):
        rows, ctr = build_rows(
            paths,
            label=lbl,
            source="spamassassin",
            base=args.spamassassin_root,
            min_chars=args.min_chars,
            debug_sample=(args.debug_sample if args.debug else 0),
            ctype_sample=(args.ctype_sample if args.debug else 0),
        )
        sa_rows.extend(rows)
        for field in (
            "files_seen",
            "archives_skipped",
            "candidates",
            "extracted_ok",
            "empty_after_parse",
            "too_short",
            "exceptions",
        ):
            setattr(
                sa_ctr_total, field, getattr(sa_ctr_total, field) + getattr(ctr, field)
            )
        sa_ctr_total.ctype_counter.update(ctr.ctype_counter)

    pd.DataFrame(sa_rows).to_csv(out_dir / "spamassassin.csv", index=False)

    # Nazario (all phishing)
    naz_rows, naz_ctr = build_rows(
        iter_message_files(args.nazario_root),
        label="phishing",
        source="nazario",
        base=args.nazario_root,
        min_chars=args.min_chars,
        debug_sample=(args.debug_sample if args.debug else 0),
        ctype_sample=(args.ctype_sample if args.debug else 0),
    )
    pd.DataFrame(naz_rows).to_csv(out_dir / "nazario.csv", index=False)

    # Enron maildir (safe)
    enron_rows, enron_ctr = build_rows(
        iter_message_files(args.enron_maildir),
        label="safe",
        source="enron",
        base=args.enron_maildir,
        min_chars=args.min_chars,
        debug_sample=(args.debug_sample if args.debug else 0),
        ctype_sample=(args.ctype_sample if args.debug else 0),
    )
    pd.DataFrame(enron_rows).to_csv(out_dir / "enron.csv", index=False)

    # Summaries
    logger.info("--- Summary: SpamAssassin ---")
    logger.info("%s", sa_ctr_total)
    logger.info("--- Summary: Nazario ---")
    logger.info("%s", naz_ctr)
    logger.info("--- Summary: Enron ---")
    logger.info("%s", enron_ctr)

    print(
        "Wrote:",
        out_dir / "spamassassin.csv",
        out_dir / "nazario.csv",
        out_dir / "enron.csv",
    )
    print("Counts (>= min chars):")
    print("  SpamAssassin:", len(sa_rows))
    print("  Nazario:     ", len(naz_rows))
    print("  Enron:       ", len(enron_rows))
    logger.info("=== Normalization done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
