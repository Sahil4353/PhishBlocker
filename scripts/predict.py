#!/usr/bin/env python
"""
Load a saved joblib model and score text(s). Prints label, probs, and top token reasons.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix

TOP_K = 6


def _feature_names(pipeline) -> List[str]:
    # Get feature names from FeatureUnion of two TfidfVectorizers
    feats = []
    fu = pipeline.named_steps["features"]
    for name, vect in fu.transformer_list:
        names = vect.get_feature_names_out()
        feats.extend([f"{name}:{n}" for n in names])
    return feats


def _vectorize(pipeline, text: str) -> csr_matrix:
    return pipeline.named_steps["features"].transform([text])


def _class_contrib(
    pipeline, x_vec: csr_matrix, class_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    clf = pipeline.named_steps["clf"]
    # For LR: contribution approximated as coef_[class] * tfidf_value
    coefs = clf.coef_[class_idx]  # shape [n_features]
    contrib = x_vec.multiply(coefs)  # sparse
    contrib = contrib.toarray().ravel()
    idx = contrib.argsort()[::-1]
    return idx, contrib


def _top_reasons(pipeline, text: str, class_idx: int, k: int = TOP_K) -> List[dict]:
    x = _vectorize(pipeline, text)
    idx_sorted, contrib = _class_contrib(pipeline, x, class_idx)
    fnames = _feature_names(pipeline)
    # keep only nonzero contributions and tokens present
    reasons = []
    taken = 0
    for i in idx_sorted:
        if contrib[i] <= 0:
            break
        tok = fnames[i]
        # strip the prefix (w_tfidf:/c_tfidf:) for UI niceness
        tok = tok.split(":", 1)[1] if ":" in tok else tok
        reasons.append({"token": tok, "weight": float(contrib[i])})
        taken += 1
        if taken >= k:
            break
    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        help="Path to joblib (e.g., models/tfidf_lr_v2.joblib)",
    )
    ap.add_argument("texts", nargs="*", help="Texts to score; omit to read from stdin")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]
    le = bundle["label_encoder"]
    classes = list(le.classes_)

    if not args.texts:
        print("Enter text (Ctrl+D to end):")
        import sys

        texts = [sys.stdin.read()]
    else:
        texts = args.texts

    for t in texts:
        probs = pipe.predict_proba([t])[0]
        pred_idx = int(np.argmax(probs))
        label = classes[pred_idx]
        reasons = _top_reasons(pipe, t, pred_idx, TOP_K)

        print("—" * 60)
        print(f"Predicted: {label}")
        print("Probabilities:")
        for c, p in zip(classes, probs):
            print(f"  {c:10s} : {p:.3f}")
        print("Top reasons:")
        for r in reasons:
            print(f"  {r['token']!r}  (+{r['weight']:.4f})")


if __name__ == "__main__":
    main()
