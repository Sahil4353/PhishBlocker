from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix

TOP_K = 6


def _unwrap_lr(est: Any) -> Any:
    # sourcery skip: assign-if-exp, reintroduce-else
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
      - predict_with_explanations(text, meta) -> (label, prob, reasons, decision_meta)
      - predict_proba_map(text) -> {label: prob, ...}

    Now supports per-class probability thresholds loaded from metrics JSON.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        version: Optional[str] = None,
        metrics_path: Optional[str | Path] = None,
    ):
        self._artifact_path = str(artifact_path)
        bundle = joblib.load(self._artifact_path)

        # Accept either {"pipeline": ..., "label_encoder": ...} or a plain pipeline.
        if isinstance(bundle, dict) and "pipeline" in bundle:
            pipe_obj = bundle["pipeline"]
            le_obj = bundle.get("label_encoder", None)
        else:
            pipe_obj = bundle
            le_obj = None

        # Explicitly annotate so static analyzers don't think these are dicts.
        self._pipe: Any = pipe_obj
        self._le: Any = le_obj

        # Class labels
        if self._le is not None and hasattr(self._le, "classes_"):
            self._classes: List[str] = list(self._le.classes_)
        else:
            self._classes = list(getattr(self._pipe, "classes_", [])) or [
                "phishing",
                "spam",
                "safe",
            ]

        # Version string for audit (used in Scan.model_version)
        self.version = version or Path(self._artifact_path).stem

        # Cache feature names for explanation
        self._feat_names: List[str] = self._compute_feature_names()

        # Thresholds (per-class min prob needed to "claim" that class)
        # Shape: {"phishing": 0.72, "spam": 0.4, ...}
        self._thresholds: Dict[str, float] = {}
        if metrics_path:
            self._thresholds = self._load_thresholds(metrics_path)

    # ---------- threshold / metrics helpers ----------

    def _load_thresholds(self, metrics_path: str | Path) -> Dict[str, float]:
        # sourcery skip: use-named-expression
        """
        Load per-class thresholds from the model metrics JSON.
        We'll look for a block like:
            {
              "threshold_suggestions": {
                "phishing": {"prec_at_95": 0.72, "chosen": 0.72},
                "spam": {"prec_at_95": 0.91, "chosen": 0.91}
              }
            }

        We'll prefer "chosen", else fallback to highest available numeric in that entry.
        If file missing/unreadable/unexpected, we just return {} and fallback to max-prob.
        """
        path = Path(metrics_path)
        if not path.exists():
            return {}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        raw = data.get("threshold_suggestions", {})
        out: Dict[str, float] = {}
        for cls_name, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            # Pick "chosen" if present, else best numeric
            if "chosen" in entry and isinstance(entry["chosen"], (int, float)):
                out[cls_name] = float(entry["chosen"])
                continue
            # fallback: take max numeric value
            numeric_vals = [
                float(v) for v in entry.values() if isinstance(v, (int, float))
            ]
            if numeric_vals:
                out[cls_name] = max(numeric_vals)
        return out

    def _apply_thresholds(self, probs: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        # sourcery skip: use-named-expression
        """
        Given raw probs[cls_i], choose a final label using per-class thresholds.
        Returns (final_label, final_prob, decision_meta)

        decision_meta will include:
        - "raw_top": {"label": <raw argmax>, "prob": <max_prob>}
        - "thresholds_used": {class: tau, ...}
        - "winner": {"label": <after-threshold label>, "prob": <prob>}
        - "fallback_reason": str
        """
        # safety: align classes
        classes = self._classes
        raw_idx = int(np.argmax(probs))
        raw_label = classes[raw_idx] if raw_idx < len(classes) else str(raw_idx)
        raw_prob = float(probs[raw_idx])

        # If we have no thresholds at all → just return argmax
        if not self._thresholds:
            return (
                raw_label,
                raw_prob,
                {
                    "raw_top": {"label": raw_label, "prob": raw_prob},
                    "thresholds_used": {},
                    "winner": {"label": raw_label, "prob": raw_prob},
                    "fallback_reason": "no_thresholds_configured",
                },
            )

        # Strategy:
        # 1. Build list of (label, prob, passes_threshold?)
        scored = []
        for i, cls_name in enumerate(classes):
            cls_prob = float(probs[i])
            tau = float(self._thresholds.get(cls_name, 0.0))
            passed = cls_prob >= tau
            scored.append(
                {
                    "label": cls_name,
                    "prob": cls_prob,
                    "tau": tau,
                    "passed": passed,
                }
            )

        # 2. Among those that passed, choose the one with max prob
        passed_candidates = [s for s in scored if s["passed"]]
        if passed_candidates:
            passed_candidates.sort(key=lambda s: s["prob"], reverse=True)
            best = passed_candidates[0]
            return (
                best["label"],
                best["prob"],
                {
                    "raw_top": {"label": raw_label, "prob": raw_prob},
                    "thresholds_used": {s["label"]: s["tau"] for s in scored},
                    "winner": {"label": best["label"], "prob": best["prob"]},
                    "fallback_reason": "passed_threshold",
                },
            )

        # 3. If nobody passed their threshold, fallback: pick raw argmax
        return (
            raw_label,
            raw_prob,
            {
                "raw_top": {"label": raw_label, "prob": raw_prob},
                "thresholds_used": {s["label"]: s["tau"] for s in scored},
                "winner": {"label": raw_label, "prob": raw_prob},
                "fallback_reason": "no_class_met_threshold",
            },
        )

    # ---------- feature name / vec helpers ----------

    def _compute_feature_names(self) -> List[str]:
        # sourcery skip: use-contextlib-suppress
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
            vect = self._find_vectorizer(self._pipe)
            if vect is not None and hasattr(vect, "get_feature_names_out"):
                try:
                    return list(vect.get_feature_names_out())
                except Exception:
                    pass
            return []

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

    def _find_vectorizer(self, obj: Any):  # sourcery skip: use-named-expression
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

        contrib = x.multiply(w).toarray().ravel()
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
    ) -> Tuple[str, float, List[Dict], Dict[str, Any]]:
        """
        Returns:
          (final_label, final_prob, reasons[], decision_meta)

        - final_label/final_prob are AFTER applying per-class thresholds (if any)
        - reasons[] are the top LR features supporting the *raw* predicted class
          (we keep this behavior the same for now)
        - decision_meta is structured info about thresholding and fallback
        """
        if not hasattr(self._pipe, "predict_proba"):
            raise RuntimeError(
                "Underlying pipeline has no predict_proba(). Use a probabilistic classifier."
            )

        probs = self._pipe.predict_proba([text])[0]  # np.ndarray
        # raw top class index
        raw_idx = int(np.argmax(probs))
        # explanations come from the raw top class (still most "influential" for that decision)
        reasons = self._top_reasons(text, raw_idx, TOP_K)

        final_label, final_prob, decision_meta = self._apply_thresholds(probs)
        return final_label, float(final_prob), reasons, decision_meta

    def predict_proba_map(self, text: str) -> Dict[str, float]:
        probs = self._pipe.predict_proba([text])[0]
        return {self._classes[i]: float(p) for i, p in enumerate(probs)}
