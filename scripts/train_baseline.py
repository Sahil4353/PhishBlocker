#!/usr/bin/env python
"""
Train TF‑IDF (word+char) + Logistic Regression baseline.
Saves a single joblib bundle with pipeline + metadata and metrics PNG/JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
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

# ---------- helpers ----------
CANON_LABELS = {
    "ham": "safe",
    "ok": "safe",
    "legit": "safe",
    "spam": "spam",
    "phish": "phishing",
    "phishing": "phishing",
    "fraud": "phishing",
}


def _canonicalize_label(x: str) -> str:
    if x is None:
        return None
    s = str(x).strip().lower()
    return CANON_LABELS.get(s, s)


def load_concat(inputs: List[str]) -> pd.DataFrame:
    frames, sha256s = [], []
    for p in inputs:
        df = pd.read_csv(p)
        if "body_text" not in df.columns:
            raise SystemExit(f"[ERR] {p} missing required column body_text")
        # common column names we’ll respect if present
        if "label" not in df.columns:
            raise SystemExit(f"[ERR] {p} missing required column label")
        df["label"] = df["label"].map(_canonicalize_label)
        frames.append(df[["body_text", "label"]].copy())
        sha256s.append(hashlib.sha256(Path(p).read_bytes()).hexdigest())
    out = pd.concat(frames, axis=0, ignore_index=True)
    out["label"] = out["label"].astype(str)
    out["body_text"] = out["body_text"].astype(str).str.strip()
    out = out[
        (out["body_text"] != "") & (out["label"].isin({"safe", "spam", "phishing"}))
    ]
    return out, sha256s


def build_vectorizer(max_features_word: int, max_features_char: int) -> FeatureUnion:
    word_tfidf = (
        "w_tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_features=max_features_word,
            lowercase=True,
            strip_accents="unicode",
        ),
    )
    char_tfidf = (
        "c_tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=max_features_char,
            lowercase=False,
        ),
    )
    return FeatureUnion([word_tfidf, char_tfidf])


def build_lr(seed: int = 42):
    return LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
        verbose=1,
    )


def build_nb():
    # NB is fast; TF-IDF works fine in practice for a baseline
    return MultinomialNB(alpha=0.1)


def build_svm_calibrated():
    base = LinearSVC(dual="auto", class_weight="balanced")
    # Platt scaling for probabilities
    return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def build_pipeline_single(
    max_features_word: int, max_features_char: int, seed: int
) -> Pipeline:
    features = build_vectorizer(max_features_word, max_features_char)
    clf = build_lr(seed)
    return Pipeline([("features", features), ("clf", clf)])


def build_pipeline_ensemble(
    max_features_word: int, max_features_char: int, seed: int
) -> Pipeline:
    features = build_vectorizer(max_features_word, max_features_char)

    lr = build_lr(seed)
    nb = build_nb()
    svm = build_svm_calibrated()

    ensemble = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("nb", nb),
            ("svm", svm),
        ],
        voting="soft",
        weights=[2.0, 1.0, 1.5],  # tweak after seeing metrics
        n_jobs=-1,
        flatten_transform=True,
    )
    return Pipeline([("features", features), ("clf", ensemble)])


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
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_pr(
    y_true_bin: np.ndarray, y_scores: np.ndarray, out_png: str, positive_label: str
) -> None:
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


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV paths")
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-features-word", type=int, default=200_000)
    ap.add_argument("--max-features-char", type=int, default=200_000)
    ap.add_argument(
        "--out", required=True, help="Joblib path, e.g., models/tfidf_lr_v1.joblib"
    )
    ap.add_argument(
        "--ensemble",
        action="store_true",
        help="Use LR+NB+calibrated LinearSVM soft-voting ensemble",
    )

    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[*] Loading data…")
    df, sha256s = load_concat(args.inputs)
    print(
        f"    Total rows after cleaning: {len(df)} "
        f"(class counts: {df['label'].value_counts().to_dict()})"
    )

    X = df["body_text"].values
    le = LabelEncoder().fit(df["label"])
    y = le.transform(df["label"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed, stratify=y
    )

    print("[*] Building pipeline…")
    if args.ensemble:
        pipe = build_pipeline_ensemble(
            args.max_features_word, args.max_features_char, args.seed
        )
    else:
        pipe = build_pipeline_single(
            args.max_features_word, args.max_features_char, args.seed
        )

    print(f"    Pipeline: {pipe}")

    print("[*] Training…")
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    train_secs = time.time() - t0
    print(f"    Done in {train_secs:.1f}s")

    print("[*] Evaluating…")
    y_pred = pipe.predict(X_te)
    y_proba = getattr(pipe.named_steps["clf"], "predict_proba", None)
    if y_proba is not None:
        proba = pipe.predict_proba(X_te)
    else:
        # In case someone swaps model w/o proba, fall back to decision_function
        dfun = pipe.decision_function(X_te)
        # min-max normalize to pseudo-probabilities per class
        dfun = (dfun - dfun.min(axis=1, keepdims=True)) / (
            dfun.ptp(axis=1, keepdims=True) + 1e-9
        )
        proba = dfun

    classes = list(le.classes_)
    report = classification_report(y_te, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_te, y_pred)

    # plots
    cm_png = out_path.with_suffix(".cm.png")
    plot_confusion(cm, classes, str(cm_png))

    # PR for phishing vs rest
    if "phishing" in classes and y_proba is not None:
        pos_idx = classes.index("phishing")
        y_true_bin = (y_te == pos_idx).astype(int)
        pr_png = out_path.with_suffix(".pr.png")
        plot_pr(y_true_bin, proba[:, pos_idx], str(pr_png), "phishing")
    else:
        pr_png = None

    # metadata
    meta: Dict = {
        "model_name": "tfidf_word+char_ensemble"
        if args.ensemble
        else "tfidf_word+char_lr",
        "version": "v2",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "scikit_learn_version": sklearn.__version__,  # joblib version too
        "params": {
            "word_ngram": [1, 2],
            "char_ngram": [3, 5],
            "max_features_word": args.max_features_word,
            "max_features_char": args.max_features_char,
            "class_weight": "balanced",
            "val_size": args.val_size,
            "seed": args.seed,
        },
        "data": {
            "inputs": args.inputs,
            "sha256": sha256s,
            "train_size": int(len(X_tr)),
            "val_size": int(len(X_te)),
            "class_distribution": df["label"].value_counts().to_dict(),
        },
        "metrics": report,
        "artifacts": {
            "confusion_png": str(cm_png),
            "pr_png": str(pr_png) if pr_png else None,
        },
        "classes": classes,
    }

    if args.ensemble:
        meta["ensemble"] = {
            "estimators": ["lr", "nb", "svm(calibrated)"],
            "weights": [2.0, 1.0, 1.5],
            "calibration": {"svm": "Platt(sigmoid)", "cv": 3},
        }

    # save bundle
    print(f"[*] Saving artifact → {out_path}")
    bundle = {
        "pipeline": pipe,
        "label_encoder": le,
        "metadata": meta,
    }
    joblib.dump(bundle, out_path)

    metrics_json = out_path.with_suffix(".metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] Saved metrics → {metrics_json}")
    print("[✓] Done.")


if __name__ == "__main__":
    main()
