#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Iterable, Tuple, List
import pandas as pd
from typing import Union
from email import policy
import email


# Try to use your project parser; fallback to a local extractor if import path differs
BytesLike = Union[bytes, bytearray, memoryview]


def _extract_text_default(raw: Union[BytesLike, str]) -> str:
    """
    Safe parser that works for bytes/bytearray/memoryview/str.
    Prefers text/plain; falls back to empty string on errors.
    """
    try:
        if isinstance(raw, (bytes, bytearray, memoryview)):
            msg = email.message_from_bytes(raw, policy=policy.default)
        else:
            # raw is str here
            msg = email.message_from_string(raw, policy=policy.default)

        parts = []
        if msg.is_multipart():
            for p in msg.walk():
                if p.get_content_type() == "text/plain":
                    # get_content() handles decoding under policy=default
                    parts.append(p.get_content() or "")
        else:
            if msg.get_content_type() == "text/plain":
                parts.append(msg.get_content() or "")
        return "\n".join(parts).strip()
    except Exception:
        return ""


def _extract_text_with_app(raw: Union[BytesLike, str]) -> str:
    """
    Try project’s parser first; fall back to default.
    """
    try:
        # Adjust import if your parser lives elsewhere
        from app.services.parser import extract_body_text  # type: ignore

        txt = extract_body_text(raw)  # your function may accept bytes or str
        if not isinstance(txt, str):
            txt = str(txt or "")
        return txt
    except Exception:
        return _extract_text_default(raw)

def iter_files(root: Path, exts: Tuple[str, ...] = (".eml", ".txt")) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def load_text(p: Path) -> str:
    try:
        raw = p.read_bytes()
    except Exception:
        return ""
    return _extract_text_with_app(raw)


def build_rows(
    paths: Iterable[Path], label: str, source: str, base: Path, min_chars: int
) -> List[dict]:
    rows = []
    for p in paths:
        text = load_text(p)
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
    # Map SpamAssassin folders → labels
    mapping = []
    for name in ["easy_ham", "easy_ham_2", "hard_ham"]:
        d = sa_root / name
        if d.exists():
            mapping.append((iter_files(d), "safe"))
    for name in ["spam", "spam_2"]:
        d = sa_root / name
        if d.exists():
            mapping.append((iter_files(d), "spam"))
    return mapping


def main():
    ap = argparse.ArgumentParser(description="Normalize EML/maildir -> canonical CSVs")
    ap.add_argument("--spamassassin-root", type=Path, required=True)
    ap.add_argument("--nazario-root", type=Path, required=True)
    ap.add_argument("--enron-maildir", type=Path, required=True)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    out_dir = args.out_dir
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
        iter_files(args.nazario_root),
        label="phishing",
        source="nazario",
        base=args.nazario_root,
        min_chars=args.min_chars,
    )
    pd.DataFrame(naz_rows).to_csv(out_dir / "nazario.csv", index=False)

    # Enron maildir (safe)
    enron_rows = build_rows(
        iter_files(args.enron_maildir),
        label="safe",
        source="enron",
        base=args.enron_maildir,
        min_chars=args.min_chars,
    )
    pd.DataFrame(enron_rows).to_csv(out_dir / "enron.csv", index=False)

    print(
        "Wrote:",
        out_dir / "spamassassin.csv",
        out_dir / "nazario.csv",
        out_dir / "enron.csv",
    )


if __name__ == "__main__":
    sys.exit(main())
    
# python scripts/datasets/normalize_eml.py `
#   --spamassassin-root data/raw/spamassassin `
#   --nazario-root      data/raw/nazario `
#   --enron-maildir     data/raw/enron/maildir `
#   --min-chars 20 `
#   --out-dir           data/processed