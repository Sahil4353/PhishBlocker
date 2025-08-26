#!/usr/bin/env python
"""
Train TF-IDF (word+char) + Logistic Regression baseline (or optional LR+NB+calibrated LinearSVM ensemble).
Saves a single joblib bundle with pipeline + metadata and metrics PNG/JSON.

Improvements:
- Robust CSV/Parquet loading with pyarrow/python engine fallbacks (+ chunked CSV)
- Structured logging with timings and defensive try/except
- Optional fast LR mode (L2/lbfgs) for large datasets
- Convergence/fit diagnostics
- Probability/PR curve handling + threshold suggestions (best-F1 and target precision)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # safe for headless runners
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
    level = (level or "INFO").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        level = "INFO"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # show ConvergenceWarning as WARNING once
    warnings.simplefilter("once", ConvergenceWarning)


log = logging.getLogger("train_baseline")

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


# ------------------------- resilient I/O -------------------------
def _hash_file(pth: Path) -> str:
    try:
        return hashlib.sha256(pth.read_bytes()).hexdigest()
    except Exception:
        return "NA"


def _read_csv_resilient(path: Path, usecols=("body_text", "label")) -> pd.DataFrame:
    """
    Try fast pyarrow engine first; if that fails:
      - Fall back to python engine with on_bad_lines='skip', chunked.
    """
    # 1) Fast path: pyarrow CSV
    try:
        df = pd.read_csv(
            path,
            usecols=list(usecols),
            dtype=str,
            engine="pyarrow",
        )
        log.info("[io] read_csv(pyarrow) rows=%d file=%s", len(df), path)
        return df
    except Exception as e:
        log.warning("[io] pyarrow CSV failed for %s: %s", path, e)

    # 2) Robust path: python engine, skip bad lines, chunked
    chunks: List[pd.DataFrame] = []
    total = 0
    try:
        for chunk in pd.read_csv(
            path,
            usecols=list(usecols),
            dtype=str,
            engine="python",
            sep=",",
            quotechar='"',
            doublequote=True,
            escapechar="\\",
            on_bad_lines="skip",
            chunksize=200_000,
        ):
            total += len(chunk)
            chunks.append(chunk)
    except Exception as e:
        raise SystemExit(f"[ERR] Robust CSV read also failed for {path}: {e}") from e

    if not chunks:
        raise SystemExit(f"[ERR] No rows loaded from {path}")
    df = pd.concat(chunks, ignore_index=True)
    log.info("[io] read_csv(python, chunks) rows=%d file=%s", total, path)
    return df


def _read_table_any(path: Path) -> pd.DataFrame:
    """Read Parquet if .parquet, else try CSV (pyarrow/python)."""
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path, engine="pyarrow")
            log.info("[io] read_parquet(pyarrow) rows=%d file=%s", len(df), path)
            return df
        except Exception as e:
            raise SystemExit(f"[ERR] Failed to read parquet {path}: {e}") from e
    else:
        return _read_csv_resilient(path)


# ------------------------- data loading -------------------------
def load_concat(inputs: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    t0 = time.perf_counter()
    frames, sha256s = [], []
    for p in inputs:
        pth = Path(p)
        if not pth.exists():
            raise SystemExit(f"[ERR] Input not found: {p}")

        sha256s.append(_hash_file(pth))

        df = _read_table_any(pth)

        for col in ("body_text", "label"):
            if col not in df.columns:
                raise SystemExit(f"[ERR] {p} missing required column: {col}")

        df = df[["body_text", "label"]].copy()
        df["body_text"] = df["body_text"].astype(str).str.strip()
        df["label"] = df["label"].map(_canonicalize_label).astype(str)
        frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=True)

    # drop empties and unknown labels
    before = len(out)
    out = out[
        (out["body_text"] != "") & (out["label"].isin({"safe", "spam", "phishing"}))
    ].copy()
    dropped = before - len(out)
    if dropped:
        log.info("[clean] Dropped %d rows (empty text or unknown label).", dropped)

    if len(out) == 0:
        raise SystemExit("[ERR] No rows left after cleaning. Check your inputs.")

    log.info("[*] Loaded %d rows (elapsed %.2fs)", len(out), time.perf_counter() - t0)
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


def build_lr_elasticnet(
    seed: int = 42, C: float = 0.5, max_iter: int = 5000
) -> LogisticRegression:
    """Original strong baseline: SAGA + elastic-net."""
    return LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.05,
        C=C,
        tol=1e-4,
        max_iter=max_iter,
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
        verbose=1,
    )


def build_lr_fast_l2(
    seed: int = 42, C: float = 1.0, max_iter: int = 1000
) -> LogisticRegression:
    """Faster option: L2 + LBFGS (recommended for very large datasets)."""
    return LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=C,
        max_iter=max_iter,
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
        verbose=0,
    )


def build_nb() -> MultinomialNB:
    return MultinomialNB(alpha=0.1)


def build_svm_calibrated() -> CalibratedClassifierCV:
    base = LinearSVC(dual="auto", class_weight="balanced")
    return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def build_pipeline_single(
    max_features_word: int,
    max_features_char: int,
    seed: int,
    fast_l2: bool,
    C: float,
    max_iter: int,
) -> Pipeline:
    features = build_vectorizer(max_features_word, max_features_char)
    clf = (
        build_lr_fast_l2(seed, C=C, max_iter=max_iter)
        if fast_l2
        else build_lr_elasticnet(seed, C=C, max_iter=max_iter)
    )
    cache_dir = Path("models/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Pipeline([("features", features), ("clf", clf)], memory=cache_dir)


def build_pipeline_ensemble(
    max_features_word: int,
    max_features_char: int,
    seed: int,
    fast_l2: bool,
    C: float,
    max_iter: int,
) -> Pipeline:
    features = build_vectorizer(max_features_word, max_features_char)
    lr = (
        build_lr_fast_l2(seed, C=C, max_iter=max_iter)
        if fast_l2
        else build_lr_elasticnet(seed, C=C, max_iter=max_iter)
    )
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
def _safe_savefig(fig: matplotlib.figure.Figure, out_png: str) -> None:
    try:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
    except Exception as e:
        log.warning("[plot] Failed to save %s: %s", out_png, e)


def plot_confusion(cm: np.ndarray, classes: List[str], out_png: str | None) -> None:
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
    if out_png:
        _safe_savefig(fig, out_png)
    plt.close(fig)


def plot_pr(
    y_true_bin: np.ndarray,
    y_scores: np.ndarray,
    out_png: str | None,
    positive_label: str,
) -> float:
    precision, recall, _ = precision_recall_curve(y_true_bin, y_scores)
    ap = average_precision_score(y_true_bin, y_scores)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    ax.step(recall, precision, where="post")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve ({positive_label}) — AP={ap:.3f}")
    fig.tight_layout()
    if out_png:
        _safe_savefig(fig, out_png)
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
    log.info(
        "[env] Python=%s | sklearn=%s | numpy=%s | pandas=%s",
        platform.python_version(),
        sklearn.__version__,
        np.__version__,
        pd.__version__,
    )
    log.info(
        "[env] Host=%s | PID=%s | CWD=%s",
        socket.gethostname(),
        os.getpid(),
        os.getcwd(),
    )
    start_all = time.perf_counter()

    try:
        # Load & clean
        log.info("[*] Loading data…")
        df, sha256s = load_concat(args.inputs)
        class_counts = df["label"].value_counts().to_dict()
        log.info(
            "    Total rows after cleaning: %d (class counts: %s)",
            len(df),
            class_counts,
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
        log.info("[split] train=%d val=%d", len(X_tr), len(X_te))

        # Build pipeline
        log.info(
            "[*] Building pipeline… (fast_l2=%s ensemble=%s)",
            args.fast_l2,
            args.ensemble,
        )
        if args.ensemble:
            pipe = build_pipeline_ensemble(
                args.max_features_word,
                args.max_features_char,
                args.seed,
                args.fast_l2,
                args.C,
                args.max_iter,
            )
        else:
            pipe = build_pipeline_single(
                args.max_features_word,
                args.max_features_char,
                args.seed,
                args.fast_l2,
                args.C,
                args.max_iter,
            )
        log.debug("    Pipeline: %s", pipe)

        # Train
        log.info("[*] Training…")
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always", ConvergenceWarning)
            pipe.fit(X_tr, y_tr)
            for w in wlist:
                if issubclass(w.category, ConvergenceWarning):
                    log.warning("ConvergenceWarning: %s", str(w.message))
        train_secs = time.perf_counter() - t0
        log.info("    Training done in %.1fs", train_secs)

        # Convergence diagnostics
        try:
            clf = pipe.named_steps["clf"]
            if hasattr(clf, "n_iter_"):
                log.info("    n_iter_: %s", getattr(clf, "n_iter_"))
            elif hasattr(clf, "estimators_"):
                for name, est in zip([e[0] for e in clf.estimators], clf.estimators_):
                    if hasattr(est, "n_iter_"):
                        log.info("    %s.n_iter_: %s", name, getattr(est, "n_iter_"))
        except Exception as e:
            log.debug("    (n_iter_ not available: %s)", e)

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
        report = classification_report(
            y_te, y_pred, target_names=classes, output_dict=True
        )
        cm = confusion_matrix(y_te, y_pred)

        # Artifacts
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cm_png = None if args.no_plots else out_path.with_suffix(".cm.png")
        if not args.no_plots:
            plot_confusion(cm, classes, str(cm_png))

        # PR curves + thresholds
        thr_suggestions: Dict[str, Dict] = {}
        ap_scores: Dict[str, float] = {}
        pr_pngs: Dict[str, str] = {}
        if has_proba and not args.no_plots:
            for label_name in ("phishing", "spam"):
                if label_name in classes:
                    idx = classes.index(label_name)
                    y_bin = (y_te == idx).astype(int)
                    pr_png = out_path.with_suffix(f".pr.{label_name}.png")
                    ap = plot_pr(y_bin, proba[:, idx], str(pr_png), label_name)
                    ap_scores[label_name] = ap
                    pr_pngs[label_name] = str(pr_png)
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
                        "prec_98": threshold_for_min_precision(
                            y_bin, proba[:, idx], 0.98
                        ),
                        "prec_95": threshold_for_min_precision(
                            y_bin, proba[:, idx], 0.95
                        ),
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
            "model_name": (
                "tfidf_word+char_ensemble"
                if args.ensemble
                else "tfidf_word+char_lr_fast_l2"
            )
            if args.fast_l2
            else (
                "tfidf_word+char_ensemble" if args.ensemble else "tfidf_word+char_lr"
            ),
            "version": "v2",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "scikit_learn_version": sklearn.__version__,
            "params": {
                "word_ngram": [1, 2],
                "char_ngram": [3, 5],
                "max_features_word": min(args.max_features_word, 150_000),
                "max_features_char": min(args.max_features_char, 150_000),
                "class_weight": "balanced",
                "val_size": args.val_size,
                "seed": args.seed,
                "fast_l2": bool(args.fast_l2),
                "C": args.C,
                "max_iter": args.max_iter,
                "lr_penalty": ("l2" if args.fast_l2 else "elasticnet"),
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
                "confusion_png": (str(cm_png) if cm_png else None),
                "pr_pngs": pr_pngs,
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

        log.info("[✓] Done. Total time: %.1fs", time.perf_counter() - start_all)
        return 0

    except KeyboardInterrupt:
        log.error("Interrupted by user (KeyboardInterrupt).")
        return 130
    except SystemExit as e:
        # Bubble up intentional SystemExit with message already logged/raised
        raise
    except Exception as e:
        log.error("Uncaught error: %s", e)
        log.debug("Traceback:\n%s", traceback.format_exc())
        return 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input files (.csv or .parquet). Must contain body_text,label",
    )
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-features-word", type=int, default=200_000)
    ap.add_argument("--max-features-char", type=int, default=200_000)
    ap.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for LR (default 1.0)",
    )
    ap.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Max iterations for LR solver (default 1000 for fast_l2)",
    )
    ap.add_argument(
        "--out", required=True, help="Joblib path, e.g., models/tfidf_lr_v2.joblib"
    )
    ap.add_argument(
        "--ensemble",
        action="store_true",
        help="Use LR+NB+calibrated LinearSVM soft-voting ensemble",
    )
    ap.add_argument(
        "--fast-l2",
        dest="fast_l2",
        action="store_true",
        help="Use faster LR (L2/lbfgs). Disable to use original SAGA+elastic-net.",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable PNG plots (confusion matrix, PR curves)",
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
    except SystemExit:
        raise
    except Exception as e:
        # Ultimate guard (shouldn't normally hit because run() handles)
        if not log.handlers:
            setup_logging("ERROR")
        log.error("Fatal error in main: %s", e)
        log.debug("Traceback:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
