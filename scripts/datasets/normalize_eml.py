#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from html import unescape
from pathlib import Path
from typing import Iterable, Tuple, List, Union

import email
import pandas as pd
from email import policy
from email.message import Message

# ----------------------------
# Bytes/str safe extractors
# ----------------------------

BytesLike = Union[bytes, bytearray, memoryview]


def _strip_html(html: str) -> str:
    # Minimal HTML → text (no external deps)
    # 1) remove scripts/styles
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    # 2) drop all tags
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    # 3) unescape entities and collapse whitespace
    html = unescape(html)
    html = re.sub(r"\s+", " ", html).strip()
    return html


def _best_text_from_message(msg: Message) -> str:
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
                ctype = (part.get_content_type() or "").lower()
                if ctype == "text/plain":
                    plain_parts.append(part.get_content() or "")
                elif ctype == "text/html":
                    html_parts.append(part.get_content() or "")
            if plain_parts:
                return "\n".join(p.strip() for p in plain_parts if p).strip()
            if html_parts:
                html = "\n".join(h for h in html_parts if h)
                return _strip_html(html)
            # fallback: any text/* part
            any_text: List[str] = []
            for part in msg.walk():
                ctype = (part.get_content_type() or "").lower()
                if ctype.startswith("text/"):
                    any_text.append(part.get_content() or "")
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
    except Exception:
        return ""


def _extract_text_default(raw: Union[BytesLike, str]) -> str:
    """
    Safe parser that works for bytes/bytearray/memoryview/str.
    """
    try:
        if isinstance(raw, (bytes, bytearray, memoryview)):
            msg = email.message_from_bytes(raw, policy=policy.default)
        else:
            msg = email.message_from_string(raw, policy=policy.default)
        return _best_text_from_message(msg)
    except Exception:
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
    except Exception:
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
    Yield likely message files:
    - include files with .eml, .txt, NO extension, or trailing-dot names (e.g., '311.')
    - skip archives and obvious non-files
    """
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if is_archive(p):
            continue
        ext = p.suffix.lower()
        if ext in (".eml", ".txt", ".", ""):
            yield p


# ----------------------------
# Core normalization
# ----------------------------

def load_text(p: Path) -> str:
    try:
        raw = p.read_bytes()
    except Exception:
        return ""
    return _extract_text_with_app(raw)


def build_rows(
    paths: Iterable[Path], label: str, source: str, base: Path, min_chars: int
) -> List[dict]:
    rows: List[dict] = []
    for p in paths:
        text = load_text(p)
        if not text:
            continue
        text = text.replace("\r\n", "\n").strip()
        if len(text) < min_chars:
            continue
        rows.append(
            {
                "body_text": text,
                "label": label,
                "source": source,
                "relpath": str(p.relative_to(base)),
                "len_chars": len(text),
            }
        )
    return rows


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
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # SpamAssassin
    sa_rows: List[dict] = []
    for paths, lbl in spamassassin_sets(args.spamassassin_root):
        sa_rows.extend(
            build_rows(
                paths,
                label=lbl,
                source="spamassassin",
                base=args.spamassassin_root,
                min_chars=args.min_chars,
            )
        )
    pd.DataFrame(sa_rows).to_csv(out_dir / "spamassassin.csv", index=False)

    # Nazario (all phishing)
    naz_rows = build_rows(
        iter_message_files(args.nazario_root),
        label="phishing",
        source="nazario",
        base=args.nazario_root,
        min_chars=args.min_chars,
    )
    pd.DataFrame(naz_rows).to_csv(out_dir / "nazario.csv", index=False)

    # Enron maildir (safe)
    enron_rows = build_rows(
        iter_message_files(args.enron_maildir),
        label="safe",
        source="enron",
        base=args.enron_maildir,
        min_chars=args.min_chars,
    )
    pd.DataFrame(enron_rows).to_csv(out_dir / "enron.csv", index=False)

    print("Wrote:", out_dir / "spamassassin.csv", out_dir / "nazario.csv", out_dir / "enron.csv")
    print("Counts (>= min chars):")
    print("  SpamAssassin:", len(sa_rows))
    print("  Nazario:     ", len(naz_rows))
    print("  Enron:       ", len(enron_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# python scripts/datasets/normalize_eml.py `
#   --spamassassin-root data/raw/spamassassin `
#   --nazario-root      data/raw/nazario `
#   --enron-maildir     data/raw/enron/maildir `
#   --min-chars 5 `
#   --out-dir           data/processed
