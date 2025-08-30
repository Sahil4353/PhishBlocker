#!/usr/bin/env python
"""
TF-IDF + (PyTorch) multinomial Logistic Regression
- Sparse CSR -> dense per-sample on the fly (no giant toarray()).
- Mini-batches, CUDA, mixed precision, class weights / weighted sampling.
- Saves torch state_dict + sklearn artifacts + metrics JSON.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

log = logging.getLogger("train_baseline")


# ------------------------- logging -------------------------
def setup_logging(level: str = "INFO") -> None:
    level = level.upper()
    level = level if level in {"DEBUG", "INFO", "WARNING", "ERROR"} else "INFO"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


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


def _canonicalize_label(x: str) -> Optional[str]:
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


# ------------------------- model -------------------------
class TorchLogReg(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


# ------------------------- dataset (CSR -> dense per item) -------------------------
class CSRDataset(Dataset):
    def __init__(self, X_csr, y_np: np.ndarray):
        self.X = X_csr
        self.y = y_np.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        # Pull a single sparse row and convert to dense float32
        row = self.X[idx]
        # row is 1 x D; convert efficiently without building the whole matrix
        x = torch.from_numpy(row.toarray().astype(np.float32, copy=False)).squeeze(0)
        y = int(self.y[idx])
        return x, y


# ------------------------- training loop -------------------------
def train_epoch(model, loader, optimizer, criterion, device, use_amp, accum_steps):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, (xb, yb) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=True)
        yb = torch.as_tensor(yb, device=device)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach().item()

    # flush remaining grads
    if (step % accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, classes: List[str]):
    model.eval()
    all_preds, all_true = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(np.asarray(yb))
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    acc = (y_pred == y_true).mean().item()
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    return acc, report, cm


# ------------------------- main -------------------------
def run(args: argparse.Namespace) -> int:
    setup_logging(args.log_level)
    t0 = time.time()

    log.info("[*] Loading data…")
    df, sha256s = load_concat(args.inputs)
    class_counts = df["label"].value_counts().to_dict()
    log.info("Rows: %d | Class counts: %s", len(df), class_counts)

    X_text = df["body_text"].values
    le = LabelEncoder().fit(df["label"])
    y = le.transform(df["label"])
    classes = list(le.classes_)

    X_tr_text, X_te_text, y_tr, y_te = train_test_split(
        X_text, y, test_size=args.val_size, random_state=args.seed, stratify=y
    )

    log.info("[*] Building TF-IDF (max_features=%d)…", args.max_features_word)
    vect = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=args.max_features_word,
        sublinear_tf=True,
        stop_words="english",
        dtype=np.float32,
    )
    X_tr = vect.fit_transform(X_tr_text)  # CSR
    X_te = vect.transform(X_te_text)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    log.info("Device: %s | AMP: %s | pin_memory: %s", device, args.mixed_precision, pin)

    # Dataset / Loader
    ds_tr = CSRDataset(X_tr, y_tr)
    ds_te = CSRDataset(X_te, y_te)

    # Optional weighted sampling (helps on imbalance)
    if args.weighted_sampler:
        class_sample_count = np.bincount(y_tr, minlength=len(classes))
        weights = 1.0 / np.clip(class_sample_count, 1, None)
        sample_weights = weights[y_tr]
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        tr_loader = DataLoader(
            ds_tr,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=pin,
            persistent_workers=False,
        )
    else:
        tr_loader = DataLoader(
            ds_tr,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin,
            persistent_workers=False,
        )

    te_loader = DataLoader(
        ds_te,
        batch_size=max(256, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=False,
    )

    # Model
    input_dim = X_tr.shape[1]
    model = TorchLogReg(input_dim, len(classes)).to(device)

    # Class weights for CE (another way to handle imbalance)
    weight_t = None
    if args.class_weight == "balanced":
        binc = np.bincount(y_tr, minlength=len(classes)).astype(np.float32)
        w = (len(y_tr) / np.clip(binc, 1, None)) / len(classes)
        weight_t = torch.tensor(w, device=device, dtype=torch.float32)
        log.info("Class weights: %s", w.tolist())

    criterion = nn.CrossEntropyLoss(weight=weight_t)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    log.info(
        "[*] Training… (epochs=%d, batch_size=%d, accum=%d)",
        args.epochs,
        args.batch_size,
        args.accum_steps,
    )
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            tr_loader,
            optimizer,
            criterion,
            device,
            args.mixed_precision,
            args.accum_steps,
        )
        acc, _, _ = evaluate(model, te_loader, device, classes)
        log.info(
            "Epoch %d/%d | loss=%.5f | val_acc=%.4f", epoch, args.epochs, loss, acc
        )

    acc, report, cm = evaluate(model, te_loader, device, classes)
    log.info("[✓] Final val_acc=%.4f", acc)

    # Save bundle
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_state": model.state_dict(),
        "input_dim": input_dim,
        "num_classes": len(classes),
        "vectorizer": vect,  # pickled inside torch.save
        "label_encoder": le,  # pickled
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "scikit_learn_version": sklearn.__version__,
            "torch_version": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "device": str(device),
            "classes": classes,
            "class_distribution": class_counts,
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
            "params": vars(args),
        },
    }
    torch.save(bundle, out_path)
    with open(out_path.with_suffix(".metrics.json"), "w", encoding="utf-8") as f:
        json.dump(bundle["metadata"], f, indent=2)

    log.info("[✓] Saved model → %s", out_path)
    log.info("[✓] Saved metrics → %s", out_path.with_suffix(".metrics.json"))
    log.info("[✓] Done in %.1fs", time.time() - t0)
    return 0


# ------------------------- CLI -------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="CSV paths with columns: body_text,label",
    )
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-features-word", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--accum-steps", type=int, default=1, help="Gradient accumulation steps"
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    ap.add_argument(
        "--weighted-sampler",
        action="store_true",
        help="Enable WeightedRandomSampler on train loader",
    )
    ap.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    ap.add_argument(
        "--no-mixed-precision", dest="mixed_precision", action="store_false"
    )
    ap.set_defaults(mixed_precision=True)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--out", required=True, help="Output .pt path (e.g., models/tfidf_lr_torch.pt)"
    )
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def main():
    try:
        args = parse_args()
        setup_logging(args.log_level)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        sys.exit(run(args))
    except Exception as e:
        log.exception("Uncaught error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
