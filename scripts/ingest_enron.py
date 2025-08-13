import argparse
import csv

try:
    csv.field_size_limit(2**31 - 1)  # ~2.1 GB
except OverflowError:
    csv.field_size_limit(2147483647)

import hashlib
import json
import re
import sys
from email import policy
from email.parser import BytesParser
from pathlib import Path

from bs4 import BeautifulSoup

URL_RE = re.compile(r"https?://[^\s)>\]\"']+", re.IGNORECASE)


def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style"]):
        t.extract()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    return [m.group(0).strip(").,>]}\"'") for m in URL_RE.finditer(text)]


def part_is_attachment(p) -> bool:
    cd = p.get_content_disposition()
    return bool(cd == "attachment" or p.get_filename())


def short_hash(*vals) -> str:
    h = hashlib.sha256()
    for v in vals:
        if not isinstance(v, (bytes, bytearray)):
            v = str(v).encode("utf-8", errors="ignore")
        h.update(v)
        h.update(b"\x00")
    return h.hexdigest()[:16]


def parse_raw(raw_bytes: bytes):
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    subject = msg.get("Subject", "") or ""
    sender = (msg.get("From", "") or "").strip()
    to = msg.get("To", "") or ""
    cc = msg.get("Cc", "") or ""
    bcc = msg.get("Bcc", "") or ""
    date = msg.get("Date", "") or ""
    text_plain, html_raw = "", ""
    attachments = []

    if msg.is_multipart():
        for p in msg.walk():
            if part_is_attachment(p):
                attachments.append(p.get_filename() or p.get_content_type())
                continue
            ctype = p.get_content_type()
            try:
                payload = p.get_content()
            except Exception:
                payload = (p.get_payload(decode=True) or b"").decode(
                    "utf-8", errors="ignore"
                )
            if ctype == "text/plain" and not text_plain:
                text_plain = str(payload)
            elif ctype == "text/html" and not html_raw:
                html_raw = str(payload)
    else:
        ctype = msg.get_content_type()
        try:
            payload = msg.get_content()
        except Exception:
            payload = (msg.get_payload(decode=True) or b"").decode(
                "utf-8", errors="ignore"
            )
        if ctype == "text/plain":
            text_plain = str(payload)
        elif ctype == "text/html":
            html_raw = str(payload)
        else:
            text_plain = str(payload)

    body_text = text_plain or html_to_text(html_raw)
    urls = list(set(extract_urls(body_text) + extract_urls(html_raw)))
    return {
        "subject": subject,
        "sender": sender,
        "to": to,
        "cc": cc,
        "bcc": bcc,
        "timestamp": date,
        "body_text": body_text,
        "html_raw": html_raw,
        "urls": urls,
        "attachments": [a for a in attachments if a],
        "attachments_count": len([a for a in attachments if a]),
    }


def main():
    ap = argparse.ArgumentParser(description="Parse Enron emails into normalized CSV.")
    ap.add_argument(
        "--input", required=True, help="Path to CSV (e.g., data/raw/enron/emails.csv)"
    )
    ap.add_argument(
        "--out", default="data/processed/enron_parsed.csv", help="Output CSV path"
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Row limit for quick test runs"
    )
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with inp.open("rb") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample.decode("utf-8", errors="ignore"))
    except Exception:
        dialect = csv.excel

    rows = []
    with inp.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        raw_col = (
            "message"
            if "message" in reader.fieldnames
            else ("text" if "text" in reader.fieldnames else None)
        )
        if not raw_col:
            print(
                f"ERROR: expected raw column 'message' (or 'text'). Found: {reader.fieldnames}"
            )
            sys.exit(1)

        for i, r in enumerate(reader):
            if args.limit and i >= args.limit:
                break
            raw = r.get(raw_col, "")
            if not raw:
                continue
            parsed = parse_raw(raw.encode("utf-8", errors="ignore"))
            uid = short_hash(
                r.get("file", ""),
                parsed["subject"],
                parsed["timestamp"],
                parsed["sender"],
            )
            rows.append(
                {
                    "id": uid,
                    "source": "enron",
                    "label": "ham",
                    "subject": parsed["subject"],
                    "sender": parsed["sender"],
                    "recipients": ";".join(
                        [parsed["to"], parsed["cc"], parsed["bcc"]]
                    ).strip(";"),
                    "body_text": parsed["body_text"],
                    "html_raw": parsed["html_raw"],
                    "timestamp": parsed["timestamp"],
                    "urls": json.dumps(parsed["urls"], ensure_ascii=False),
                    "attachments": json.dumps(
                        parsed["attachments"], ensure_ascii=False
                    ),
                    "attachments_count": parsed["attachments_count"],
                }
            )

    if not rows:
        print("No rows parsed. Check input path/format.")
        sys.exit(1)

    cols = [
        "id",
        "source",
        "label",
        "subject",
        "sender",
        "recipients",
        "body_text",
        "html_raw",
        "timestamp",
        "urls",
        "attachments",
        "attachments_count",
    ]
    with outp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows → {outp}")


if __name__ == "__main__":
    sys.exit(main())
