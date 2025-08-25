#!/usr/bin/env python
"""
Train TF-IDF (word+char) + Logistic Regression baseline (optional LR+NB+calibrated LinearSVM ensemble).
Saves a single joblib bundle with pipeline + metadata and metrics PNG/JSON.

This version adds:
- Structured logging
- Validations (required columns, ≥2 classes, non-empty texts)
- Graceful error handling
- Convergence/fit diagnostics
- Probability/PR curve handling + threshold suggestions (best-F1 and target precision)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import csr_matrix  # noqa: F401 (kept for type hints / future use)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


# ------------------------- logging helpers -------------------------
def setup_logging(level: str = "INFO") -> None:
    level = level.upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        level = "INFO"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # show ConvergenceWarning as WARNING once
    warnings.simplefilter("once", ConvergenceWarning)


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
        # keep only what we use
        df = df[["body_text", "label"]].copy()
        frames.append(df)
        try:
            sha256s.append(hashlib.sha256(pth.read_bytes()).hexdigest())
        except Exception:
            sha256s.append("NA")

    out = pd.concat(frames, axis=0, ignore_index=True)

    out["body_text"] = out["body_text"].astype(str).str.strip()
    out["label"] = out["label"].astype(str)

    # drop empties and unknown labels
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


# ------------------------- model builders -------------------------
def build_vectorizer(max_features_word: int, max_features_char: int) -> FeatureUnion:
    """Word(1–2) + char(3–5) TF-IDF; lean defaults to keep RAM/time in check."""
    word_tfidf = (
        "w_tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            max_features=min(max_features_word, 150_000),
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            dtype=np.float32,
        ),
    )
    char_tfidf = (
        "c_tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=3,
            max_features=min(max_features_char, 150_000),
            lowercase=False,
            dtype=np.float32,
        ),
    )
    return FeatureUnion([word_tfidf, char_tfidf])


def build_lr(seed: int = 42) -> LogisticRegression:
    """Strong, calibrated-ish linear baseline. Elastic-net for mild sparsity; C tweaked for precision."""
    return LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.05,
        C=0.5,
        tol=1e-4,
        max_iter=5000,
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
        verbose=1,
    )


def build_nb() -> MultinomialNB:
    return MultinomialNB(alpha=0.1)


def build_svm_calibrated() -> CalibratedClassifierCV:
    base = LinearSVC(dual="auto", class_weight="balanced")
    return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def build_pipeline_single(
    max_features_word: int, max_features_char: int, seed: int
) -> Pipeline:
    features = build_vectorizer(max_features_word, max_features_char)
    clf = build_lr(seed)
    cache_dir = Path("models/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Pipeline([("features", features), ("clf", clf)], memory=cache_dir)


def build_pipeline_ensemble(
    max_features_word: int, max_features_char: int, seed: int
) -> Pipeline:
    features = build_vectorizer(max_features_word, max_features_char)
    lr = build_lr(seed)
    nb = build_nb()
    svm = build_svm_calibrated()
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("nb", nb), ("svm", svm)],
        voting="soft",
        weights=[2.0, 1.0, 1.5],  # tune later if needed
        n_jobs=-1,
        flatten_transform=True,
    )
    cache_dir = Path("models/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Pipeline([("features", features), ("clf", ensemble)], memory=cache_dir)


# ------------------------- plotting helpers -------------------------
def plot_confusion(cm: np.ndarray, classes: List[str], out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_pr(
    y_true_bin: np.ndarray, y_scores: np.ndarray, out_png: str, positive_label: str
) -> float:
    precision, recall, _ = precision_recall_curve(y_true_bin, y_scores)
    ap = average_precision_score(y_true_bin, y_scores)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    ax.step(recall, precision, where="post")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve ({positive_label}) — AP={ap:.3f}")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return float(ap)


# ------------------------- threshold suggestions -------------------------
def best_threshold_by_f1(
    y_true_bin: np.ndarray, y_scores: np.ndarray
) -> Tuple[float, float, float, float]:
    p, r, th = precision_recall_curve(y_true_bin, y_scores)
    f1 = (2 * p * r) / (p + r + 1e-12)
    j = int(np.argmax(f1))
    # Note: precision_recall_curve returns len(th) = len(p) - 1
    t = float(th[j]) if j < len(th) else 0.5
    return t, float(p[j]), float(r[j]), float(f1[j])


def threshold_for_min_precision(
    y_true_bin: np.ndarray, y_scores: np.ndarray, min_precision: float
) -> Dict[str, float | bool]:
    p, r, th = precision_recall_curve(y_true_bin, y_scores)
    idx = np.where(p >= min_precision)[0]
    if len(idx) == 0:
        t, P, R, F = best_threshold_by_f1(y_true_bin, y_scores)
        return {
            "target_precision": min_precision,
            "available": False,
            "threshold": t,
            "P": P,
            "R": R,
            "F1": F,
        }
    j = int(idx[0])
    # Map p/r index to threshold index
    t = 0.0 if j == 0 else float(th[j - 1])
    return {
        "target_precision": min_precision,
        "available": True,
        "threshold": t,
        "P": float(p[j]),
        "R": float(r[j]),
    }


# ------------------------- main -------------------------
def run(args: argparse.Namespace) -> int:
    setup_logging(args.log_level)
    start_all = time.time()

    # Load & clean
    log.info("[*] Loading data…")
    df, sha256s = load_concat(args.inputs)
    class_counts = df["label"].value_counts().to_dict()
    log.info(
        "    Total rows after cleaning: %d (class counts: %s)", len(df), class_counts
    )

    # Encode labels
    X = df["body_text"].values
    le = LabelEncoder().fit(df["label"])
    y = le.transform(df["label"])
    classes = list(le.classes_)

    # Validate class count
    if len(np.unique(y)) < 2:
        raise SystemExit(
            "[ERR] Need at least 2 classes to train. Provide spam/phishing rows in inputs."
        )

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed, stratify=y
    )

    # Build pipeline
    log.info("[*] Building pipeline…")
    if args.ensemble:
        pipe = build_pipeline_ensemble(
            args.max_features_word, args.max_features_char, args.seed
        )
    else:
        pipe = build_pipeline_single(
            args.max_features_word, args.max_features_char, args.seed
        )
    log.debug("    Pipeline: %s", pipe)

    # Train
    log.info("[*] Training…")
    t0 = time.time()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(X_tr, y_tr)
        for w in wlist:
            if issubclass(w.category, ConvergenceWarning):
                log.warning("ConvergenceWarning: %s", str(w.message))
    train_secs = time.time() - t0
    log.info("    Done in %.1fs", train_secs)

    # Convergence diagnostics
    try:
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "n_iter_"):
            log.info("    n_iter_: %s", getattr(clf, "n_iter_"))
        elif hasattr(clf, "estimators_"):
            for name, est in zip([e[0] for e in clf.estimators], clf.estimators_):
                if hasattr(est, "n_iter_"):
                    log.info("    %s.n_iter_: %s", name, getattr(est, "n_iter_"))
    except Exception:
        log.debug("    (n_iter_ not available)")

    # Evaluate
    log.info("[*] Evaluating…")
    y_pred = pipe.predict(X_te)

    # Handle probabilities (LR/NB/calibrated SVM expose predict_proba)
    has_proba = hasattr(pipe, "predict_proba")
    if has_proba:
        proba = pipe.predict_proba(X_te)
    else:
        # Decision function → min-max normalize row-wise
        dfun = pipe.decision_function(X_te)
        dfun = (dfun - dfun.min(axis=1, keepdims=True)) / (
            dfun.ptp(axis=1, keepdims=True) + 1e-9
        )
        proba = dfun

    # Metrics
    report = classification_report(y_te, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_te, y_pred)

    # Artifacts
    out_path = Path(args.out)
    cm_png = out_path.with_suffix(".cm.png")
    plot_confusion(cm, classes, str(cm_png))

    # PR curves + thresholds
    thr_suggestions: Dict[str, Dict] = {}
    ap_scores: Dict[str, float] = {}
    if has_proba:
        for label_name in ("phishing", "spam"):
            if label_name in classes:
                idx = classes.index(label_name)
                y_bin = (y_te == idx).astype(int)
                pr_png = out_path.with_suffix(f".pr.{label_name}.png")
                ap = plot_pr(y_bin, proba[:, idx], str(pr_png), label_name)
                ap_scores[label_name] = ap
                t_best, p_best, r_best, f_best = best_threshold_by_f1(
                    y_bin, proba[:, idx]
                )
                thr_suggestions[label_name] = {
                    "best_f1": {
                        "threshold": t_best,
                        "P": p_best,
                        "R": r_best,
                        "F1": f_best,
                    },
                    "prec_98": threshold_for_min_precision(y_bin, proba[:, idx], 0.98),
                    "prec_95": threshold_for_min_precision(y_bin, proba[:, idx], 0.95),
                }
                log.info(
                    "    %s AP=%.3f | bestF1@t=%.3f (P=%.3f, R=%.3f, F1=%.3f)",
                    label_name,
                    ap,
                    t_best,
                    p_best,
                    r_best,
                    f_best,
                )

    # Metadata
    meta: Dict = {
        "model_name": "tfidf_word+char_ensemble"
        if args.ensemble
        else "tfidf_word+char_lr",
        "version": "v2",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "params": {
            "word_ngram": [1, 2],
            "char_ngram": [3, 5],
            "max_features_word": min(args.max_features_word, 150_000),
            "max_features_char": min(args.max_features_char, 150_000),
            "class_weight": "balanced",
            "val_size": args.val_size,
            "seed": args.seed,
            "lr_penalty": "elasticnet",
            "l1_ratio": 0.05,
            "C": 0.5,
            "tol": 1e-4,
            "max_iter": 5000,
        },
        "data": {
            "inputs": args.inputs,
            "sha256": sha256s,
            "train_size": int(len(X_tr)),
            "val_size": int(len(X_te)),
            "class_distribution": class_counts,
        },
        "metrics": report,
        "artifacts": {
            "confusion_png": str(cm_png),
            "pr_pngs": {
                k: str(out_path.with_suffix(f".pr.{k}.png")) for k in ap_scores.keys()
            },
        },
        "classes": classes,
        "threshold_suggestions": thr_suggestions,
        "average_precision": ap_scores,
    }

    if args.ensemble:
        meta["ensemble"] = {
            "estimators": ["lr", "nb", "svm(calibrated)"],
            "weights": [2.0, 1.0, 1.5],
            "calibration": {"svm": "Platt(sigmoid)", "cv": 3},
        }

    # Save bundle + metrics
    log.info("[*] Saving artifact → %s", out_path)
    bundle = {"pipeline": pipe, "label_encoder": le, "metadata": meta}
    try:
        joblib.dump(bundle, out_path)
    except Exception as e:
        log.error("Failed to save model bundle to %s: %s", out_path, e)
        return 2

    metrics_json = out_path.with_suffix(".metrics.json")
    try:
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        log.info("[✓] Saved metrics → %s", metrics_json)
    except Exception as e:
        log.error("Failed to write metrics JSON to %s: %s", metrics_json, e)
        return 2

    log.info("[✓] Done. Total time: %.1fs", time.time() - start_all)
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV paths")
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-features-word", type=int, default=200_000)
    ap.add_argument("--max-features-char", type=int, default=200_000)
    ap.add_argument(
        "--out", required=True, help="Joblib path, e.g., models/tfidf_lr_v2.joblib"
    )
    ap.add_argument(
        "--ensemble",
        action="store_true",
        help="Use LR+NB+calibrated LinearSVM soft-voting ensemble",
    )
    ap.add_argument(
        "--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR (default: INFO)"
    )
    return ap.parse_args()


def main() -> None:
    try:
        args = parse_args()
        code = run(args)
        sys.exit(code)
    except SystemExit as e:
        # SystemExit raised intentionally with a message
        raise
    except Exception as e:
        log = logging.getLogger(__name__)
        if not log.handlers:
            setup_logging("ERROR")
        log.error("Uncaught error: %s", e)
        log.debug("Traceback:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
