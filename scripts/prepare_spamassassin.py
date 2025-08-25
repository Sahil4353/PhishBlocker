import csv
import email
import glob
import hashlib
import os
import re
import sys
from email import policy
from email.parser import BytesParser

BASE = r"data/raw/spamassassin"
OUT = r"data/processed/spamassassin_parsed.csv"
rows = []


def norm(txt):
    if txt is None:
        return ""
    # collapse whitespace; drop null bytes
    return re.sub(r"\s+", " ", txt.replace("\x00", " ")).strip()


def extract_body(msg):
    if msg.is_multipart():
        # prefer text/plain
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    return part.get_content().strip()
                except:
                    pass
        # fallback: join all text
        parts = []
        for part in msg.walk():
            if part.get_content_type().startswith("text/"):
                try:
                    parts.append(part.get_content())
                except:
                    pass
        return "\n".join(parts)
    else:
        try:
            return msg.get_content().strip()
        except:
            return msg.get_payload(decode=True) or ""


def add_dir(path, label):
    for fp in glob.glob(os.path.join(path, "*")):
        if os.path.isdir(fp):
            continue
        try:
            with open(fp, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)
            body = norm(extract_body(msg))
            if not body:
                continue
            rid = hashlib.sha256(fp.encode()).hexdigest()[:16]
            rows.append(
                [rid, "spamassassin", label, "", "", "", body, "", "", "[]", "[]", 0]
            )
        except Exception as e:
            # skip unreadables
            pass


add_dir(os.path.join(BASE, "easy_ham"), "ham")
add_dir(os.path.join(BASE, "hard_ham"), "ham")
add_dir(os.path.join(BASE, "spam"), "spam")
add_dir(os.path.join(BASE, "spam_2"), "spam")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(
        [
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
    )
    for r in rows:
        w.writerow(r)

print(f"Wrote {len(rows)} rows to {OUT}")
