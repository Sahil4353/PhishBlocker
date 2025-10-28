from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix

TOP_K = 6


def _unwrap_lr(est: Any) -> Any:
    """
    Try to unwrap a LogisticRegression from common wrappers like CalibratedClassifierCV or Pipeline.
    Returns the estimator if it has a `coef_`, else None.
    """
    # direct LR
    if hasattr(est, "coef_"):
        return est
    # CalibratedClassifierCV / similar
    base = getattr(est, "base_estimator", None) or getattr(est, "estimator", None)
    if base is not None and hasattr(base, "coef_"):
        return base
    return None


class ModelService:
    """
    Wraps a trained text classification pipeline (typically TF-IDF + LR).
    Exposes:
      - predict_with_explanations(text, meta) -> (label, prob, reasons[])
      - predict_proba_map(text) -> {label: prob, ...}
    """

    def __init__(self, artifact_path: str | Path, version: Optional[str] = None):
        self._artifact_path = str(artifact_path)
        bundle = joblib.load(self._artifact_path)

        # Accept either {"pipeline": ..., "label_encoder": ...} or a plain pipeline.
        if isinstance(bundle, dict) and "pipeline" in bundle:
            self._pipe = bundle["pipeline"]
            self._le = bundle.get("label_encoder", None)
        else:
            self._pipe = bundle
            self._le = None

        # Class labels
        if self._le is not None and hasattr(self._le, "classes_"):
            self._classes: List[str] = list(self._le.classes_)
        else:
            self._classes = list(getattr(self._pipe, "classes_", [])) or [
                "phishing",
                "spam",
                "safe",
            ]

        # Version string for audit
        # Prefer provided version, else filename stem.
        self.version = version or Path(self._artifact_path).stem

        # cache feature names for explanation
        self._feat_names: List[str] = self._compute_feature_names()

    # ---------- internal helpers ----------

    def _compute_feature_names(self) -> List[str]:
        """
        Try to recover human-readable feature names from the vectorizer(s).
        Supports:
        - a 'features' step that is a FeatureUnion / ColumnTransformer
        - or a single vectorizer in the pipeline
        """
        names_out: List[str] = []
        try:
            fu = self._pipe.named_steps.get("features")
        except Exception:
            fu = None

        if fu is None:
            # Try whole pipeline for a vectorizer
            vect = self._find_vectorizer(self._pipe)
            if vect is not None and hasattr(vect, "get_feature_names_out"):
                try:
                    return list(vect.get_feature_names_out())
                except Exception:
                    pass
            return []

        # FeatureUnion / ColumnTransformer path:
        trans_list = getattr(fu, "transformer_list", None)
        if not trans_list:
            vect = self._find_vectorizer(fu)
            if vect is not None and hasattr(vect, "get_feature_names_out"):
                try:
                    return list(vect.get_feature_names_out())
                except Exception:
                    return []
            return []

        for name, transformer, *rest in trans_list:
            vect = self._find_vectorizer(transformer)
            if vect is not None and hasattr(vect, "get_feature_names_out"):
                try:
                    feats = list(vect.get_feature_names_out())
                    names_out.extend([f"{name}:{f}" for f in feats])
                except Exception:
                    continue
        return names_out

    def _find_vectorizer(self, obj: Any):
        """
        Return the last step under `obj` that exposes get_feature_names_out (e.g. TfidfVectorizer).
        """
        if obj is None:
            return None
        if hasattr(obj, "get_feature_names_out"):
            return obj

        named = getattr(obj, "named_steps", None)
        if named:
            for step_name, step in named.items():
                if hasattr(step, "get_feature_names_out"):
                    return step

        steps = getattr(obj, "steps", None)
        if steps:
            for _, step in steps[::-1]:
                if hasattr(step, "get_feature_names_out"):
                    return step
        return None

    def _vectorize(self, text: str) -> csr_matrix:
        feats = getattr(self._pipe, "named_steps", {}).get("features", None)
        if feats is None:
            # Fall back to "pipeline minus clf"
            try:
                non_clf = self._pipe[:-1]
                return non_clf.transform([text])
            except Exception as e:
                raise RuntimeError(
                    "Vectorization stage not found; ensure a 'features' step or a vectorizer before 'clf'."
                ) from e
        return feats.transform([text])

    def _get_clf_and_coefs(self) -> Tuple[Any, Optional[np.ndarray]]:
        clf = getattr(self._pipe, "named_steps", {}).get("clf", None)
        if clf is None:
            return None, None

        lr = _unwrap_lr(clf)
        if lr is not None and hasattr(lr, "coef_"):
            coefs = lr.coef_
            if coefs.ndim == 1:
                coefs = coefs.reshape(1, -1)
            return clf, coefs
        return clf, None

    def _top_reasons(self, text: str, class_idx: int, k: int = TOP_K) -> List[Dict]:
        x = self._vectorize(text)  # (1, n_features) sparse
        clf, coefs = self._get_clf_and_coefs()
        if coefs is None:
            return []

        # Binary case: coef_.shape[0] == 1 → single row of weights
        if coefs.shape[0] == 1:
            w = coefs[0]
        else:
            w = coefs[class_idx]

        contrib = x.multiply(w).toarray().ravel()  # per-feature contribution
        if contrib.size == 0:
            return []

        idx_sorted = np.argsort(contrib)[::-1]
        reasons: List[Dict] = []
        feat_names = self._feat_names if self._feat_names else [
            f"f{i}" for i in range(contrib.size)
        ]

        for i in idx_sorted:
            if contrib[i] <= 0:
                break
            tok = feat_names[i]
            if ":" in tok:
                tok = tok.split(":", 1)[1]  # drop "word:" prefix etc.
            reasons.append({"token": tok, "weight": float(contrib[i])})
            if len(reasons) >= k:
                break
        return reasons

    # ---------- public API ----------

    @property
    def classes(self) -> List[str]:
        return self._classes

    def predict_with_explanations(
        self, text: str, meta: Optional[Dict] = None
    ) -> Tuple[str, float, List[Dict]]:
        """
        Returns (label, max_prob, reasons[]).
        reasons[] is [{"token": "...", "weight": float}, ...].
        """
        if not hasattr(self._pipe, "predict_proba"):
            raise RuntimeError(
                "Underlying pipeline has no predict_proba(). Use a probabilistic classifier."
            )

        probs = self._pipe.predict_proba([text])[0]
        pred_idx = int(np.argmax(probs))
        label = (
            self._classes[pred_idx] if pred_idx < len(self._classes) else str(pred_idx)
        )

        reasons = self._top_reasons(text, pred_idx, TOP_K)
        return label, float(probs[pred_idx]), reasons

    def predict_proba_map(self, text: str) -> Dict[str, float]:
        probs = self._pipe.predict_proba([text])[0]
        return {self._classes[i]: float(p) for i, p in enumerate(probs)}