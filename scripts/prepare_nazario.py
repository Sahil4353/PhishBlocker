import csv
import hashlib
import os

import pandas as pd

SRC = r"data/raw/nazario/Nazario.csv"
OUT = r"data/processed/nazario_phishing.csv"

# Try common body column names
df = pd.read_csv(SRC)
for BODY_COL in ["body", "text", "message", "Body", "content"]:
    if BODY_COL in df.columns:
        break
else:
    raise SystemExit(
        f"Couldn't find a body column in {SRC}. Columns: {list(df.columns)}"
    )


def rid(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:16]


df["id"] = df.index.map(lambda i: rid(f"nazario-{i}"))
df["source"] = "nazario"
df["label"] = "phishing"
df["subject"] = ""
df["sender"] = ""
df["recipients"] = ""
df["body_text"] = df[BODY_COL].astype(str)
df["html_raw"] = ""
df["timestamp"] = ""
df["urls"] = "[]"
df["attachments"] = "[]"
df["attachments_count"] = 0

out = df[
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
]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.to_csv(OUT, index=False)
print(f"Wrote {len(out)} rows to {OUT}")
