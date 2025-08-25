from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix

TOP_K = 6


class ModelService:
    def __init__(self, artifact_path: str):
        self._artifact_path = artifact_path
        self._bundle = joblib.load(artifact_path)
        self._pipe = self._bundle["pipeline"]
        self._le = self._bundle["label_encoder"]
        self._classes = list(self._le.classes_)
        self._feat_names = self._compute_feature_names()

    def _compute_feature_names(self) -> List[str]:
        feats = []
        fu = self._pipe.named_steps["features"]
        for name, vect in fu.transformer_list:
            names = vect.get_feature_names_out()
            feats.extend([f"{name}:{n}" for n in names])
        return feats

    def _vectorize(self, text: str) -> csr_matrix:
        return self._pipe.named_steps["features"].transform([text])

    def _top_reasons(self, text: str, class_idx: int, k: int = TOP_K) -> List[Dict]:
        x = self._vectorize(text)
        clf = self._pipe.named_steps["clf"]
        coefs = clf.coef_[class_idx]
        contrib = x.multiply(coefs).toarray().ravel()
        idx_sorted = contrib.argsort()[::-1]

        reasons = []
        for i in idx_sorted:
            if contrib[i] <= 0:
                break
            tok = self._feat_names[i]
            tok = tok.split(":", 1)[1] if ":" in tok else tok
            reasons.append({"token": tok, "weight": float(contrib[i])})
            if len(reasons) >= k:
                break
        return reasons

    def predict_with_explanations(
        self, text: str, meta: Optional[Dict] = None
    ) -> Tuple[str, float, List[Dict]]:
        probs = self._pipe.predict_proba([text])[0]
        pred_idx = int(np.argmax(probs))
        label = self._classes[pred_idx]
        reasons = self._top_reasons(text, pred_idx, TOP_K)
        return label, float(probs[pred_idx]), reasons


# Example usage (wire later in FastAPI route):
# svc = ModelService("models/tfidf_lr_v2.joblib")
# label, prob, reasons = svc.predict_with_explanations(email_text)
