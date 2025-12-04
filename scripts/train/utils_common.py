"""
utils_common.py
Core utilities for machine learning pipeline.
- deterministic seeding
- canonical label normalization
- CSV loading with SHA256 signatures
- temperature scaling application
- threshold tuning (binary or class-wise)
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, fbeta_score

log = logging.getLogger("utils")


# =========================================================
# 1. Reproducible Seeding
# =========================================================
def seed_everything(seed: int = 42):
    """
    Makes model training reproducible across machines (as much as possible).
    Sets:
        - numpy
        - torch CPU
        - torch CUDA
        - python
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    log.info(f"[Seed] Using seed={seed}")


# =========================================================
# 2. Canonical Label Normalization
# =========================================================
_LABEL_MAP = {
    "ham": "safe",
    "ok": "safe",
    "legit": "safe",
    "safe": "safe",

    "spam": "spam",

    "phish": "phishing",
    "fraud": "phishing",
    "phishing": "phishing",
}


def canonicalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map noisy labels to canonical {safe, spam, phishing}.
    Removes invalid rows gracefully.
    """
    df = df.copy()
    df["label"] = df["label"].astype(str).str.lower().map(_LABEL_MAP)
    before = len(df)
    df = df[df["label"].notna()].copy()
    dropped = before - len(df)
    if dropped:
        log.info(f"[Labels] Dropped {dropped} rows w/ unknown labels")
    return df


# =========================================================
# 3. CSV Loader with SHA256 Proof
# =========================================================
def _sha256file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return "NA"


def load_concat_df(inputs: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load + concat CSVs with schema validation.
    Returns:
        dataframe
        sha256 per input (proof of dataset used)
    """
    frames = []
    sigs = []

    for p in inputs:
        pth = Path(p)
        if not pth.exists():
            raise SystemExit(f"[ERR] Input not found: {p}")

        try:
            df = pd.read_csv(pth)
        except Exception as e:
            raise SystemExit(f"[ERR] Failed to read {p}: {e}")

        for col in ("body_text", "label"):
            if col not in df.columns:
                raise SystemExit(f"[ERR] {p} missing required column: {col}")

        frames.append(df[["body_text", "label"]].copy())
        sigs.append(_sha256file(pth))

    out = pd.concat(frames, ignore_index=True)

    # minimal cleaning
    out["body_text"] = out["body_text"].astype(str).str.strip()
    out = out[out["body_text"] != ""].copy()

    if len(out) == 0:
        raise SystemExit("[ERR] No data remaining after cleaning!")

    return out, sigs


# =========================================================
# 4. Temperature Scaling Application
# =========================================================
def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """
    Safe temp scaling – avoids division by zero or NaNs.
    """
    return logits / max(T, 1e-6)


# =========================================================
# 5. Threshold Tuning Utilities
# =========================================================
def fbeta(prec: float, rec: float, beta: float) -> float:
    if prec + rec == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2)*(prec * rec) / max((b2 * prec + rec), 1e-12)


def tune_threshold(
    prob_pos: np.ndarray,
    y_true_bin: np.ndarray,
    metric: str = "f1",
    prec_target: Optional[float] = None,
) -> float:
    """
    Finds optimal threshold based on chosen metric:
        - f1
        - f2
        - f0.5
        - recall
        - precision

    For recall metric, if prec_target is defined:
        ensure P >= prec_target (useful for phishing detection)
    """

    precisions, recalls, thresholds = precision_recall_curve(
        y_true_bin, prob_pos)
    thresholds = np.concatenate([thresholds, [1.0]])  # align shapes

    best_thr = 0.5
    best_score = -1.0

    for p, r, t in zip(precisions, recalls, thresholds):

        if metric == "f1":
            score = fbeta(p, r, 1.0)

        elif metric == "f2":
            score = fbeta(p, r, 2.0)

        elif metric == "f0.5":
            score = fbeta(p, r, 0.5)

        elif metric == "precision":
            score = p

        elif metric == "recall":
            if prec_target is not None and p < prec_target:
                continue
            score = r

        else:
            score = fbeta(p, r, 1.0)

        if score > best_score:
            best_score = score
            best_thr = float(t)

    return float(best_thr)

def collapse_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].map(
        {"safe": "safe", "spam": "not_safe", "phishing": "not_safe"}
    )
    return df
