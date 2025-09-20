# services/model.py
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
    # CalibratedClassifierCV
    base = getattr(est, "base_estimator", None) or getattr(est, "estimator", None)
    if base is not None and hasattr(base, "coef_"):
        return base
    return None


class ModelService:
    def __init__(self, artifact_path: str | Path):
        self._artifact_path = str(artifact_path)
        bundle = joblib.load(self._artifact_path)

        # Support either {"pipeline": ..., "label_encoder": ...} or a plain pipeline
        if isinstance(bundle, dict) and "pipeline" in bundle:
            self._pipe = bundle["pipeline"]
            self._le = bundle.get("label_encoder", None)
        else:
            self._pipe = bundle
            self._le = None

        # Determine classes (label encoder preferred, else pipeline.classes_)
        if self._le is not None and hasattr(self._le, "classes_"):
            self._classes: List[str] = list(self._le.classes_)
        else:
            self._classes = list(getattr(self._pipe, "classes_", [])) or [
                "phishing",
                "spam",
                "safe",
            ]

        # Precompute feature names (best-effort; fall back to f{idx})
        self._feat_names: List[str] = self._compute_feature_names()

    # ---------- internals ----------

    def _compute_feature_names(self) -> List[str]:
        """
        Extract feature names from a typical text-features stage:
        - FeatureUnion or ColumnTransformer named 'features'
        - Each subtransformer might be a Vectorizer or a Pipeline ending in a Vectorizer
        We prefix with the transformer name to disambiguate, then strip prefix in explanations.
        """
        names_out: List[str] = []
        try:
            fu = self._pipe.named_steps.get("features")
        except Exception:
            fu = None

        if fu is None:
            # Try to get names from the whole pipeline (single vectorizer case)
            vect = self._find_vectorizer(self._pipe)
            if vect is not None and hasattr(vect, "get_feature_names_out"):
                try:
                    return list(vect.get_feature_names_out())
                except Exception:
                    pass
            # Fallback: unknown names, will be "f{i}"
            return []

        # FeatureUnion-like (FeatureUnion or ColumnTransformer share attribute names)
        trans_list = getattr(fu, "transformer_list", None)
        if not trans_list:
            # Possibly a Pipeline: try to find the last vectorizer inside
            vect = self._find_vectorizer(fu)
            if vect is not None and hasattr(vect, "get_feature_names_out"):
                try:
                    return list(vect.get_feature_names_out())
                except Exception:
                    return []
            return []

        for name, transformer, *rest in trans_list:
            # ColumnTransformer has (name, transformer, columns)
            # FeatureUnion has (name, transformer)
            vect = self._find_vectorizer(transformer)
            if vect is not None and hasattr(vect, "get_feature_names_out"):
                try:
                    feats = list(vect.get_feature_names_out())
                    names_out.extend([f"{name}:{f}" for f in feats])
                except Exception:
                    # ignore this block; continue others
                    continue
        return names_out

    def _find_vectorizer(self, obj: Any):
        """
        Return the last step in a Pipeline/obj that exposes get_feature_names_out (e.g., TfidfVectorizer).
        """
        if obj is None:
            return None
        if hasattr(obj, "get_feature_names_out"):
            return obj
        # Pipelines have named_steps / steps
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
        feats = self._pipe.named_steps.get("features", None)
        if feats is None:
            # Try whole pipeline minus classifier (assumes last step is 'clf')
            try:
                non_clf = self._pipe[:-1]
                return non_clf.transform([text])
            except Exception as e:
                raise RuntimeError(
                    "Vectorization stage not found; ensure a 'features' step or a vectorizer before 'clf'."
                ) from e
        return feats.transform([text])

    def _get_clf_and_coefs(self) -> Tuple[Any, Optional[np.ndarray]]:
        clf = self._pipe.named_steps.get("clf", None)
        if clf is None:
            return None, None
        # unwrap to LR if possible (for explanations)
        lr = _unwrap_lr(clf)
        if lr is not None and hasattr(lr, "coef_"):
            coefs = lr.coef_
            # Ensure 2D (n_classes, n_features) for uniform handling
            if coefs.ndim == 1:
                coefs = coefs.reshape(1, -1)
            return clf, coefs
        return clf, None

    def _top_reasons(self, text: str, class_idx: int, k: int = TOP_K) -> List[Dict]:
        x = self._vectorize(text)  # (1, n_features) sparse
        clf, coefs = self._get_clf_and_coefs()
        if coefs is None:
            # Explanations not available (non-linear model, no coef_, or wrapped unsupported)
            return []

        # Handle binary case where coef_.shape[0] == 1 → weights correspond to the positive class
        if coefs.shape[0] == 1:
            w = coefs[0]
        else:
            # Multi-class one-vs-rest
            w = coefs[class_idx]

        # element-wise contribution ≈ x_j * w_j
        # csr.multiply with 1-D array multiplies by columns
        contrib = x.multiply(w).toarray().ravel()  # shape (n_features,)
        if contrib.size == 0:
            return []

        idx_sorted = np.argsort(contrib)[::-1]
        reasons: List[Dict] = []
        feat_names = (
            self._feat_names
            if self._feat_names
            else [f"f{i}" for i in range(contrib.size)]
        )

        for i in idx_sorted:
            if contrib[i] <= 0:
                break
            tok = feat_names[i]
            if ":" in tok:
                tok = tok.split(":", 1)[1]
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
        Returns (label, max_prob, reasons[]) where reasons is a list of {"token","weight"}.
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
        """
        Convenience: full distribution as {class: prob}
        """
        probs = self._pipe.predict_proba([text])[0]
        return {self._classes[i]: float(p) for i, p in enumerate(probs)}
