from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import hstack


class TorchLogReg(nn.Module):
    """
    Must match the architecture used in scripts/train/train_baseline.py:

        class TorchLogReg(nn.Module):
            def __init__(self, input_dim: int, num_classes: int):
                super().__init__()
                self.linear = nn.Linear(input_dim, num_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TorchTfidfEmailModel:
    """
    Runtime wrapper around a TF-IDF + Torch logistic regression model.

    It loads:
      - a TorchLogReg checkpoint (.pt) saved by train_baseline.py
      - word and (optional) char TfidfVectorizer joblibs
      - a LabelEncoder joblib

    And exposes a sklearn-like API:
      - predict_proba(texts) -> np.ndarray [n_samples, n_classes]
      - predict(texts) -> np.ndarray of string labels
      - classes_ -> list of class labels
    """

    def __init__(
        self,
        model_path: str,
        models_dir: Optional[str] = None,
        use_char: bool = True,
    ) -> None:
        # ---------- Load checkpoint ----------
        model_path = str(model_path)
        ckpt = torch.load(model_path, map_location="cpu")

        input_dim: int = int(ckpt["input_dim"])
        num_classes: int = int(ckpt["num_classes"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TorchLogReg(input_dim, num_classes).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # ---------- Load vectorizers + label encoder ----------
        base_dir = (
            Path(models_dir) if models_dir is not None else Path(model_path).parent
        )

        # word TF-IDF is always expected
        self.vect_word = joblib.load(base_dir / "vectorizer_word.joblib")

        # char TF-IDF is optional (depends on --use-char at training time)
        char_path = base_dir / "vectorizer_char.joblib"
        self.vect_char = joblib.load(char_path) if char_path.exists() else None

        # label encoder
        self.label_encoder = joblib.load(base_dir / "label_encoder.joblib")

        # public classes_ attribute for compatibility
        self.classes_: List[str] = list(self.label_encoder.classes_)

        # only use char features if both requested and available
        self.use_char = use_char and self.vect_char is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_features(self, texts: List[str]):
        """
        Transform raw texts into feature matrix using word (+ optional char) TF-IDF.
        Returns a CSR sparse matrix.
        """
        Xw = self.vect_word.transform(texts)
        if self.use_char and self.vect_char is not None:
            Xc = self.vect_char.transform(texts)
            X = hstack([Xw, Xc], format="csr")
        else:
            X = Xw
        return X

    # ------------------------------------------------------------------
    # sklearn-like public API
    # ------------------------------------------------------------------

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        texts: list of raw email bodies (or a single string).
        Returns: numpy array of shape [n_samples, n_classes] with probabilities.
        """
        if isinstance(texts, str):
            texts = [texts]

        X = self._make_features(texts)
        X_dense = torch.from_numpy(X.toarray().astype(np.float32, copy=False))
        X_dense = X_dense.to(self.device)

        with torch.no_grad():
            logits = self.model(X_dense)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Return class labels for each text (using argmax over predict_proba).
        """
        probs = self.predict_proba(texts)
        idx = probs.argmax(axis=1)
        return self.label_encoder.inverse_transform(idx)
