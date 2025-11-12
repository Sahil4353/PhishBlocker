#!/usr/bin/env python
"""
TF-IDF + PyTorch Logistic Regression (GPU-ready)
- Word + Char TF-IDF (optional) with sparse hstack
- Mini-batch training from CSR without building giant dense matrices
- Mixed precision (AMP), class weights, optional weighted sampler
- Binary mode (safe vs not_safe) with threshold tuning
- Temperature scaling for probability calibration
- Saves: model state_dict, sklearn artifacts (joblib), metrics JSON, and plots (PR, ROC, CM)
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

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

log = logging.getLogger("train_baseline")

# --- Windows-safe globals for CSR collation (top-level so workers can pickle) ---
_GLOBAL_X_CSR: Optional[csr_matrix] = None
_GLOBAL_Y_NP: Optional[np.ndarray] = None


# ------------------------- logging -------------------------
def setup_logging(level: str = "INFO") -> None:
    level = level.upper()
    level = level if level in {"DEBUG", "INFO", "WARNING", "ERROR"} else "INFO"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def csr_worker_init(worker_id: int):
    """Runs in each worker; capture dataset's CSR pointers into globals."""
    info = torch.utils.data.get_worker_info()
    # In the main process this can be None, but as a worker_init_fn it should be set.
    if info is None:
        return

    ds = info.dataset
    # Help the type checker: we know this is always IndexCSRDataset here.
    if not isinstance(ds, IndexCSRDataset):
        raise TypeError("csr_worker_init expects an IndexCSRDataset")

    global _GLOBAL_X_CSR, _GLOBAL_Y_NP
    _GLOBAL_X_CSR = ds.X
    _GLOBAL_Y_NP = ds.y


def collate_csr_indices(batch_indices):
    """Vectorized CSR -> dense for a whole batch, using globals set in worker_init."""
    if _GLOBAL_X_CSR is None or _GLOBAL_Y_NP is None:
        raise RuntimeError("CSR globals not initialised; did csr_worker_init run?")

    idx = np.fromiter(batch_indices, dtype=np.int64)
    xb = torch.from_numpy(
        _GLOBAL_X_CSR[idx].toarray().astype(np.float32, copy=False)
    )
    yb = torch.from_numpy(_GLOBAL_Y_NP[idx])
    return xb, yb


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
BIN_MAP = {"safe": "safe", "spam": "not_safe", "phishing": "not_safe"}


def _canonicalize_label(x: str) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    return CANON_LABELS.get(s, s)


# ------------------------- IO -------------------------
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


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling for logits."""

    def __init__(self):
        super().__init__()
        self.logT = nn.Parameter(torch.zeros(1))  # T = exp(logT) >= 0

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.logT)
        return logits / T


# ------------------------- dataset (CSR -> dense per item) -------------------------
class IndexCSRDataset(Dataset):
    """Dataset that returns just indices; the collate_fn builds dense batches from CSR in one shot."""

    def __init__(self, X_csr, y_np: np.ndarray):
        self.X = X_csr
        self.y = y_np.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        # Return only the index; batching & CSR->dense happens in collate_fn
        return int(idx)


def make_loaders(
    ds_tr, ds_te, y_tr: np.ndarray, num_classes: int, args, device
) -> tuple[DataLoader, DataLoader]:
    """High-throughput DataLoaders: more CPU workers, pinned memory, prefetch."""
    import os

    pin = device.type == "cuda"
    cpu = max(1, (os.cpu_count() or 4) - 1)
    num_workers = args.num_workers if args.num_workers is not None else cpu
    if num_workers <= 0:
        num_workers = cpu

    common = dict(
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    if args.weighted_sampler:
        binc = np.bincount(y_tr, minlength=num_classes)
        weights = 1.0 / np.clip(binc, 1, None)
        sample_w = weights[y_tr]
        sampler = WeightedRandomSampler(
            sample_w.tolist(),
            num_samples=len(sample_w),
            replacement=True,
    )

        tr_loader = DataLoader(
            ds_tr,
            batch_size=args.batch_size,
            sampler=sampler,
            **{k: v for k, v in common.items() if v is not None},
        )
    else:
        tr_loader = DataLoader(
            ds_tr,
            batch_size=args.batch_size,
            shuffle=True,
            **{k: v for k, v in common.items() if v is not None},
        )

    te_loader = DataLoader(
        ds_te,
        batch_size=max(256, args.batch_size),
        shuffle=False,
        **{k: v for k, v in common.items() if v is not None},
    )
    return tr_loader, te_loader


# ------------------------- training / eval -------------------------
def train_epoch(model, loader, optimizer, criterion, device, use_amp, accum_steps):
    """Faster minibatch loop with AMP + gradient accumulation and simple throughput logging."""
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    n_seen = 0
    t0 = time.time()
    for step, (xb, yb) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=True)
        yb = torch.as_tensor(yb, device=device)

        with torch.autocast(
            device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=use_amp
        ):
            logits = model(xb)
            loss = criterion(logits, yb) / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach().item()
        n_seen += xb.size(0)

    # flush remaining grads
    if (step % accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    secs = max(1e-6, time.time() - t0)
    log.debug("epoch_throughput: %.1f samples/sec", n_seen / secs)
    return running_loss / max(1, len(loader))


@torch.no_grad()
def predict_logits(model, loader, device) -> np.ndarray:
    model.eval()
    chunks = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).detach().cpu().numpy()
        chunks.append(logits)
    return np.vstack(chunks)


@torch.no_grad()
def evaluate(
    model, loader, device, classes: List[str]
) -> Tuple[float, Dict, List[List[int]]]:
    logits = predict_logits(model, loader, device)
    preds = logits.argmax(axis=1)
    y_true = (
        np.concatenate([np.asarray(y) for _, y in loader.dataset.__iter__()])
        if False
        else None
    )  # Not used

    # Rebuild y_true from loader (safe way)
    ys = []
    for _, yb in loader:
        ys.append(np.asarray(yb))
    y_true = np.concatenate(ys)

    acc = (preds == y_true).mean().item()
    report = classification_report(
        y_true, preds, target_names=classes, output_dict=True
    )
    cm = confusion_matrix(y_true, preds).tolist()
    return acc, report, cm


def tune_temperature(
    model, loader, device, num_classes: int, max_iter: int = 200
) -> float:
    """Fits temperature (NLL on val set). Returns T as float."""
    model.eval()
    scaler = TemperatureScaler().to(device)
    optimizer = optim.LBFGS(
        scaler.parameters(), lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe"
    )
    nll = nn.CrossEntropyLoss()

    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb.to(device, non_blocking=True))
        ys.append(torch.as_tensor(yb, device=device))
    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        logits_T = scaler(logits)
        loss = nll(logits_T, Y)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(torch.exp(scaler.logT).detach().cpu().item())
    return T


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return logits / max(T, 1e-6)


# ------------------------- plotting -------------------------
def _save_plot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_confusion_matrix(cm: List[List[int]], classes: List[str], path: Path):
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center")
    ax.set_ylabel("True")
    ax.set_xlabel("Pred")
    _save_plot(path)


def plot_pr_roc(
    y_true_bin: np.ndarray, prob_pos: np.ndarray, out_pr: Path, out_roc: Path
):
    # PR
    prec, rec, thr = precision_recall_curve(y_true_bin, prob_pos)
    ap = average_precision_score(y_true_bin, prob_pos)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.title(f"Precision-Recall (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    _save_plot(out_pr)

    # ROC
    fpr, tpr, _ = roc_curve(y_true_bin, prob_pos)
    auc = roc_auc_score(y_true_bin, prob_pos)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.title(f"ROC (AUC={auc:.3f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    _save_plot(out_roc)


def make_loaders_from_csr(X_tr, y_tr, X_te, y_te, num_classes: int, args, device):
    """High-throughput DataLoaders with pinned memory and batch CSR->dense (Windows-safe)."""
    import os

    pin = device.type == "cuda"
    cpu = max(1, (os.cpu_count() or 4) - 1)
    num_workers = args.num_workers if args.num_workers is not None else cpu
    if num_workers < 0:
        num_workers = cpu

    ds_tr = IndexCSRDataset(X_tr, y_tr)
    ds_te = IndexCSRDataset(X_te, y_te)

    common = dict(
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_csr_indices,
        worker_init_fn=csr_worker_init,
    )

    if args.weighted_sampler:
        binc = np.bincount(y_tr, minlength=num_classes)
        weights = 1.0 / np.clip(binc, 1, None)
        sample_w = weights[y_tr]
        sampler = WeightedRandomSampler(
            sample_w.tolist(),
            num_samples=len(sample_w),
            replacement=True,
    )
    tr_loader = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        sampler=sampler,
        **{k: v for k, v in common.items() if v is not None},
    )
    else:
        tr_loader = DataLoader(
            ds_tr,
            batch_size=args.batch_size,
            shuffle=True,
            **{k: v for k, v in common.items() if v is not None},
        )

    te_loader = DataLoader(
        ds_te,
        batch_size=max(256, args.batch_size),
        shuffle=False,
        **{k: v for k, v in common.items() if v is not None},
    )
    return tr_loader, te_loader


# ------------------------- main -------------------------
def run(args: argparse.Namespace) -> int:
    # Performance knobs to push GPU/CPU harder
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    setup_logging(args.log_level)
    t0 = time.time()

    log.info("[*] Loading data…")
    df, sha256s = load_concat(args.inputs)

    # Optional binary collapse to emphasize recall on not_safe
    if getattr(args, "binary", False):
        df["label"] = df["label"].map(
            {"safe": "safe", "spam": "not_safe", "phishing": "not_safe"}
        )

    class_counts = df["label"].value_counts().to_dict()
    log.info("Rows: %d | Class counts: %s", len(df), class_counts)

    X_text = df["body_text"].values
    le = LabelEncoder().fit(df["label"])
    y = le.transform(df["label"])
    classes = list(le.classes_)
    num_classes = len(classes)
    log.info("Classes: %s", classes)

    # --- Guard: stratified split fails if any class has < 2 samples ---
    min_count = min(class_counts.values())
    if min_count < 2:
        log.warning(
            "Some classes have <2 samples (%s). Disabling stratified split. "
            "Consider using --binary or collecting more data for those classes.",
            class_counts,
        )
        stratify_y = None
    else:
        stratify_y = y

    X_tr_text, X_te_text, y_tr, y_te = train_test_split(
        X_text,
        y,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=stratify_y,
    )

    # --- TF-IDF stacks ---
    log.info("[*] Building word TF-IDF (max_features=%d)…", args.max_features_word)
    vect_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=args.max_features_word,
        sublinear_tf=True,
        stop_words="english",
        dtype=np.float32,
    )
    Xw_tr = vect_word.fit_transform(X_tr_text)
    Xw_te = vect_word.transform(X_te_text)

    vect_char = None
    if getattr(args, "use_char", False):
        log.info(
            "[*] Building char TF-IDF (ngram=%s, max_features=%d)…",
            str(tuple(getattr(args, "char_ngram", (3, 5)))),
            getattr(args, "max_features_char", 10000),
        )
        vect_char = TfidfVectorizer(
            analyzer="char",
            ngram_range=tuple(getattr(args, "char_ngram", (3, 5))),
            max_features=getattr(args, "max_features_char", 10000),
            sublinear_tf=True,
            dtype=np.float32,
        )
        Xc_tr = vect_char.fit_transform(X_tr_text)
        Xc_te = vect_char.transform(X_te_text)
        X_tr = hstack([Xw_tr, Xc_tr], format="csr")
        X_te = hstack([Xw_te, Xc_te], format="csr")
    else:
        X_tr, X_te = Xw_tr, Xw_te

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.mixed_precision and device.type == "cuda"
    log.info("Device: %s | AMP: %s", device, use_amp)


    # Datasets / Loaders (multiclass; batch CSR->dense for speed)
    tr_loader, te_loader = make_loaders_from_csr(
        X_tr, y_tr, X_te, y_te, num_classes, args, device
    )

    # Model
    input_dim = X_tr.shape[1]
    model = TorchLogReg(input_dim, num_classes).to(device)

    # Loss: prioritize recall on positive class by class_weight='balanced'
    weight_t = None
    if args.class_weight == "balanced":
        binc = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
        w = (len(y_tr) / np.clip(binc, 1, None)) / num_classes
        weight_t = torch.tensor(w, device=device, dtype=torch.float32)
        log.info("Class weights: %s", w.tolist())

    criterion = nn.CrossEntropyLoss(weight=weight_t)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ---- Train with "best-by-recallish" early model save ----
    best_metric = -1.0
    best_state = None

    # In binary mode, evaluate each epoch with an F2-like score on val probs.
    def _val_recallish_score(current_model) -> float:
        logits = predict_logits(current_model, te_loader, device)
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        y_val = np.concatenate([np.asarray(yb) for _, yb in te_loader])

        if getattr(args, "binary", False):
            # Recall-heavy score: F2 at a tuned threshold
            cls = np.array(classes)
            pos_idx = int(np.where(cls == "not_safe")[0][0])
            prob_pos = probs[:, pos_idx]
            thr = tune_threshold(prob_pos, (y_val == pos_idx).astype(int), metric="f2")
            y_hat = (prob_pos >= thr).astype(int)
            # compute F2
            from sklearn.metrics import fbeta_score

            return float(fbeta_score((y_val == pos_idx).astype(int), y_hat, beta=2.0))
        else:
            # multiclass macro recall
            from sklearn.metrics import recall_score

            y_pred = probs.argmax(axis=1)
            return float(recall_score(y_val, y_pred, average="macro"))

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
        use_amp,
        args.accum_steps,
    )


        # quick val metric focusing on recall
        score = _val_recallish_score(model)
        log.info(
            "Epoch %d/%d | loss=%.5f | recallish=%.4f", epoch, args.epochs, loss, score
        )

        if score > best_metric:
            best_metric = score
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            log.debug("New best checkpoint (recallish=%.4f)", best_metric)

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Optional temperature scaling
    T = None
    if getattr(args, "calibrate", False):
        log.info("[*] Calibrating probabilities (temperature scaling)…")
        T = tune_temperature(model, te_loader, device, num_classes=num_classes)
        log.info("Calibrated temperature T=%.4f", T)

    # Final eval
    logits_val = predict_logits(model, te_loader, device)
    if T is not None:
        logits_val = apply_temperature(logits_val, T)
    probs_val = torch.softmax(torch.tensor(logits_val), dim=1).numpy()
    y_val = np.concatenate([np.asarray(yb) for _, yb in te_loader])

    # Threshold tuning for binary with precision floor (good recall with “good” precision)
    tuned_threshold = None
    pr_curve_png = None
    roc_curve_png = None
    if getattr(args, "binary", False):
        pos_idx = int(np.where(np.array(classes) == "not_safe")[0][0])
        prob_pos = probs_val[:, pos_idx]
        y_true_bin = (y_val == pos_idx).astype(int)

        # prioritize recall subject to reasonable precision (0.90 default)
        precision_target = 0.90
        tuned_threshold = tune_threshold(
            prob_pos, y_true_bin, metric="recall", prec_target=precision_target
        )
        log.info(
            "Tuned threshold for recall@P>=%.2f: %.4f",
            precision_target,
            tuned_threshold,
        )

        # plots
        out_dir = Path(args.out).parent
        pr_curve_png = out_dir / "pr_curve.png"
        roc_curve_png = out_dir / "roc_curve.png"
        plot_pr_roc(y_true_bin, prob_pos, pr_curve_png, roc_curve_png)

        # ---- Compute standard metrics (post-threshold for binary if selected) ----
    if getattr(args, "binary", False) and tuned_threshold is not None:
        # Binary safe vs not_safe using tuned threshold
        pos_idx = int(np.where(np.array(classes) == "not_safe")[0][0])
        y_true_bin = (y_val == pos_idx).astype(int)
        y_pred_bin = (probs_val[:, pos_idx] >= tuned_threshold).astype(int)

        report = classification_report(
            y_true_bin,
            y_pred_bin,
            target_names=["safe", "not_safe"],
            output_dict=True,
        )
        cm = confusion_matrix(y_true_bin, y_pred_bin).tolist()
        acc = float((y_pred_bin == y_true_bin).mean())
    else:
        # Multiclass, or binary without a tuned threshold: use argmax
        y_pred = probs_val.argmax(axis=1)
        report = classification_report(
            y_val, y_pred, target_names=classes, output_dict=True
        )
        cm = confusion_matrix(y_val, y_pred).tolist()
        acc = float((y_pred == y_val).mean())
    log.info("[✓] Final val_acc=%.4f", acc)

    # ---- Save artifacts ----
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # sklearn artifacts separately (for portability)
    joblib.dump(vect_word, out_dir / "vectorizer_word.joblib")
    if vect_char is not None:
        joblib.dump(vect_char, out_dir / "vectorizer_char.joblib")
    joblib.dump(le, out_dir / "label_encoder.joblib")

    # best model state only
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
        },
        out_path,
    )

    # Save CM plot
    plot_confusion_matrix(
        cm,
        classes if not getattr(args, "binary", False) else ["safe", "not_safe"],
        out_dir / "confusion_matrix.png",
    )

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "device": str(device),
        "classes": (
            classes if not getattr(args, "binary", False) else ["safe", "not_safe"]
        ),
        "class_distribution": class_counts,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "params": vars(args),
        "sha256_inputs": sha256s,
        "temperature": T,
        "tuned_threshold": tuned_threshold,
        "plots": {
            "pr_curve": str(pr_curve_png) if pr_curve_png else None,
            "roc_curve": str(roc_curve_png) if roc_curve_png else None,
            "confusion_matrix": str(out_dir / "confusion_matrix.png"),
        },
        "early_best_metric": best_metric,
    }
    with open(out_path.with_suffix(".metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"val_acc={acc:.4f}\n")
        if tuned_threshold is not None:
            f.write(f"tuned_threshold={tuned_threshold:.4f}\n")
        if T is not None:
            f.write(f"temperature={T:.4f}\n")
        f.write(f"best_recallish={best_metric:.4f}\n")

    log.info("[✓] Saved model → %s", out_path)
    log.info("[✓] Saved metrics → %s", out_path.with_suffix(".metrics.json"))
    log.info("[✓] Artifacts dir: %s", out_dir)
    log.info("[✓] Done in %.1fs", time.time() - t0)
    return 0


# ------------------------- threshold tuning -------------------------
def fbeta(prec: float, rec: float, beta: float) -> float:
    if prec + rec == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * (prec * rec) / max(b2 * prec + rec, 1e-12)


def tune_threshold(
    prob_pos: np.ndarray,
    y_true_bin: np.ndarray,
    metric: str = "f1",
    prec_target: Optional[float] = None,
) -> float:
    # Evaluate on all unique thresholds from PR curve
    precisions, recalls, thresholds = precision_recall_curve(y_true_bin, prob_pos)
    thresholds = np.concatenate([thresholds, [1.0]])  # align lengths
    best_thr, best_score = 0.5, -1.0

    for p, r, t in zip(precisions, recalls, thresholds):
        if metric == "f1":
            score = fbeta(p, r, 1.0)
        elif metric == "f2":
            score = fbeta(p, r, 2.0)
        elif metric == "f0.5":
            score = fbeta(p, r, 0.5)
        elif metric == "recall":
            if prec_target is not None and p < prec_target:
                continue
            score = r
        elif metric == "precision":
            score = p
        else:
            score = fbeta(p, r, 1.0)
        if score > best_score:
            best_score, best_thr = score, float(t)
    return float(best_thr)


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
    ap.add_argument(
        "--use-char", action="store_true", help="Enable char TF-IDF and hstack"
    )
    ap.add_argument(
        "--char-ngram",
        nargs=2,
        type=int,
        default=[3, 5],
        help="Char ngram range, e.g., 3 5",
    )
    ap.add_argument("--max-features-char", type=int, default=10000)

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
        "--binary", action="store_true", help="Collapse labels to safe vs not_safe"
    )
    ap.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Tune decision threshold (binary only)",
    )
    ap.add_argument(
        "--tune-metric",
        choices=["f1", "f2", "f0.5", "recall", "precision"],
        default="f1",
    )
    ap.add_argument(
        "--precision-target",
        type=float,
        default=None,
        help="For metric=recall, enforce precision >= target",
    )

    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable temperature scaling on validation set",
    )

    ap.add_argument(
        "--out",
        required=True,
        help="Output model .pt path (e.g., models/tfidf_lr_torch.pt)",
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
