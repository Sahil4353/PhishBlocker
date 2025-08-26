# PhishBlocker — Data Directory

This folder holds **local datasets only**. Large files are **ignored by git** on purpose
to keep the repository small and fast to clone. Each teammate should stage datasets
locally following the structure below.

> ✅ Never commit raw/processed datasets to the repo. Use Parquet locally for speed.
> See `.gitignore` for the ignore rules.

---

## Layout

data/
├─ .gitkeep # keep the folder in git
├─ raw/ # raw sources (do not edit)
│ ├─ .gitkeep
│ ├─ enron/ # e.g., Enron dump
│ │ └─ emails.csv # ~1.3 GB (ignored by git)
│ └─ other_sources/ # any additional corpora
│
└─ processed/ # cleaned/normalized tables for training
├─ .gitkeep
├─ enron_parsed.csv # ~0.7 GB (ignored)
├─ spamassassin_parsed.csv # ~4.7k rows (ignored)
├─ nazario_phishing.csv # ~1.5k rows (ignored)
├─ enron_parsed.parquet # preferred (fast) format (ignored)
├─ spamassassin_parsed.parquet # preferred (ignored)
└─ nazario_phishing.parquet # preferred (ignored)


---

## How to place datasets

1. Download or generate datasets and place them under `data/raw/` or `data/processed/`.
2. Prefer **Parquet** (`*.parquet`) for training — loads faster and avoids CSV quoting issues.

### Convert CSV → Parquet (once per file)

```python
import pandas as pd
from pathlib import Path

csv = Path(r"data/processed/spamassassin_parsed.csv")
pq  = csv.with_suffix(".parquet")

try:
    df = pd.read_csv(csv, usecols=["body_text","label"], dtype=str, engine="pyarrow")
except Exception:
    df = pd.read_csv(csv, usecols=["body_text","label"], dtype=str, engine="python", on_bad_lines="skip")

df.to_parquet(pq, engine="pyarrow", index=False)
print("Wrote:", pq)
