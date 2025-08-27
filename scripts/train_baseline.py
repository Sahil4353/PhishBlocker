#!/usr/bin/env python
"""
PyTorch-powered TF-IDF + Logistic Regression training script.
- Uses scikit-learn for TF-IDF feature extraction.
- Uses PyTorch (with CUDA if available) for training logistic regression.
- Saves model + metrics JSON.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------- logging -------------------------
def setup_logging(level: str = "INFO") -> None:
    level = level.upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        level = "INFO"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

log = logging.getLogger(__name__)

# ------------------------- label normalization -------------------------
CANON_LABELS = {
    "ham": "safe",
    "ok": "safe",
    "legit": "safe",
    "safe": "safe",
    "spam": "spam",
    "phish": "phishing",
    "phishing": "phishing",
    "fraud": "phishing",
}

def _canonicalize_label(x: str) -> str | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    return CANON_LABELS.get(s, s)

# ------------------------- data loading -------------------------
def load_concat(inputs: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    frames, sha256s = [], []
    for p in inputs:
        pth = Path(p)
        if not pth.exists():
            raise SystemExit(f"[ERR] Input not found: {p}")
        try:
            df = pd.read_csv(p)
        except Exception as e:
            raise SystemExit(f"[ERR] Failed to read {p}: {e}") from e

        for col in ("body_text", "label"):
            if col not in df.columns:
                raise SystemExit(f"[ERR] {p} missing required column: {col}")

        df["label"] = df["label"].map(_canonicalize_label)
        df = df[["body_text", "label"]].copy()
        frames.append(df)
        try:
            sha256s.append(hashlib.sha256(pth.read_bytes()).hexdigest())
        except Exception:
            sha256s.append("NA")

    out = pd.concat(frames, axis=0, ignore_index=True)
    out["body_text"] = out["body_text"].astype(str).str.strip()
    out["label"] = out["label"].astype(str)

    before = len(out)
    out = out[
        (out["body_text"] != "") & (out["label"].isin({"safe", "spam", "phishing"}))
    ].copy()
    dropped = before - len(out)
    if dropped:
        log.info("Dropped %d rows (empty text or unknown label).", dropped)

    if len(out) == 0:
        raise SystemExit("[ERR] No rows left after cleaning. Check your inputs.")

    return out, sha256s

# ------------------------- PyTorch Model -------------------------
class TorchLogReg(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# ------------------------- main training -------------------------
def run(args: argparse.Namespace) -> int:
    setup_logging(args.log_level)
    start_all = time.time()

    log.info("[*] Loading data…")
    df, sha256s = load_concat(args.inputs)
    class_counts = df["label"].value_counts().to_dict()
    log.info("    Total rows after cleaning: %d (class counts: %s)", len(df), class_counts)

    X = df["body_text"].values
    le = LabelEncoder().fit(df["label"])
    y = le.transform(df["label"])
    classes = list(le.classes_)

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed, stratify=y
    )

    # TF-IDF
    log.info("[*] Building TF-IDF features (max_word=%d, max_char=%d)…",
             args.max_features_word, args.max_features_char)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=args.max_features_word,
        sublinear_tf=True,
        stop_words="english",
        dtype=np.float32,
    )
    X_tr_tfidf = vectorizer.fit_transform(X_tr)
    X_te_tfidf = vectorizer.transform(X_te)

    # Torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[*] Using device: {device}")

    X_train = torch.tensor(X_tr_tfidf.toarray(), dtype=torch.float32).to(device)
    y_train = torch.tensor(y_tr, dtype=torch.long).to(device)
    X_test = torch.tensor(X_te_tfidf.toarray(), dtype=torch.float32).to(device)
    y_test = torch.tensor(y_te, dtype=torch.long).to(device)

    # Model
    model = TorchLogReg(X_train.shape[1], len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
    log.info("[*] Training model…")
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            log.info(f"Epoch {epoch}/{args.epochs}, Loss: {loss.item():.4f}")

    # Evaluation
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)
        acc = (preds == y_test).float().mean().item()
        log.info(f"[✓] Accuracy: {acc:.4f}")
        report = classification_report(y_test.cpu(), preds.cpu(), target_names=classes, output_dict=True)
        cm = confusion_matrix(y_test.cpu(), preds.cpu()).tolist()

    # Save model + metadata
    out_path = Path(args.out)
    bundle = {
        "model_state": model.state_dict(),
        "vectorizer": vectorizer,
        "label_encoder": le,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "scikit_learn_version": sklearn.__version__,
            "torch_version": torch.__version__,
            "classes": classes,
            "class_distribution": class_counts,
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
            "params": vars(args),
        },
    }

    torch.save(bundle, out_path)
    metrics_json = out_path.with_suffix(".metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(bundle["metadata"], f, indent=2)

    log.info(f"[✓] Saved model → {out_path}")
    log.info(f"[✓] Saved metrics → {metrics_json}")
    log.info("[✓] Done in %.1fs", time.time() - start_all)
    return 0

# ------------------------- CLI -------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV paths")
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-features-word", type=int, default=10000)
    ap.add_argument("--max-features-char", type=int, default=2000)  # not used yet, but kept
    ap.add_argument("--epochs", type=int, default=10, help="Training epochs")
    ap.add_argument("--out", required=True, help="Output model path (e.g., models/tfidf_lr_torch.pt)")
    ap.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR (default: INFO)")
    return ap.parse_args()

def main() -> None:
    try:
        args = parse_args()
        code = run(args)
        sys.exit(code)
    except Exception as e:
        log.error("Uncaught error: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
