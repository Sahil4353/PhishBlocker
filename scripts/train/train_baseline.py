#!/usr/bin/env python
"""
train_baseline.py
Clean OOP orchestrator for PhishBlocker TF-IDF baseline model.

Features:
- CSV ingestion + hashing
- Label canonicalization
- TF-IDF (word + optional char) feature stack
- EDA plots
- DataLoaders (CSR -> dense)
- Model training with AMP + accumulation
- Early checkpointing
- Temperature scaling (post-hoc calibration)
- Full evaluation (PR/ROC, confusion, curves)
- Optional GPU profiling
- Strong metadata export

Author: You + ChatGPT refactor
"""

from __future__ import annotations

# =========================================================
# STD LIB
# =========================================================
import argparse
import logging
import json
import joblib
from pathlib import Path
from datetime import datetime, timezone

# =========================================================
# ML / PY LIBS
# =========================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================================================
# PROJECT MODULES
# =========================================================
from scripts.train.data_loader import make_loaders
from scripts.train.model import TorchLogReg, TemperatureScaler
from scripts.train.trainer import train_epoch, evaluate, predict_logits
from scripts.train.plotting_eda import run_eda
from scripts.train.plotting_ml import (
    plot_confusion_matrix,
    plot_multiclass_pr_roc,
    plot_training_curves,
)
from scripts.train.utils_common import (
    seed_everything,
    load_concat_df,
    canonicalize_labels,
    tune_threshold,
    apply_temperature,
)
from scripts.train.profiling_gpu import run_gpu_benchmark


log = logging.getLogger("pipeline")


# =========================================================
# PIPELINE CLASS
# =========================================================
class TrainerPipeline:
    """
    Encapsulates entire ML lifecycle:
    - load → prepare → train → eval → export
    """

    def __init__(self, args):
        self.args = args
        seed_everything(args.seed)
        self.out_path = Path(args.out)
        self.out_dir = self.out_path.parent
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        log.info(f"[Device] {self.device}")

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------
    def load_data(self):
        df, sha = load_concat_df(self.args.inputs)
        df = canonicalize_labels(df)

        # binary collapse
        if self.args.binary:
            df["label"] = df["label"].map({
                "safe": "safe",
                "spam": "not_safe",
                "phishing": "not_safe",
            })

        self.df = df
        self.sha = sha
        log.info(f"[Data] Loaded {len(df)} rows")

    def encode_labels(self):
        y_raw = self.df["label"].astype(str).values
        self.le = LabelEncoder().fit(y_raw)
        self.y_all = self.le.transform(y_raw)
        self.classes = list(self.le.classes_)
        self.num_classes = len(self.classes)
        log.info(f"[Classes] {self.classes}")

    def split_data(self):
        Xtxt = self.df["body_text"].astype(str).values
        self.tr_text, self.te_text, self.tr_y, self.te_y = train_test_split(
            Xtxt,
            self.y_all,
            test_size=self.args.val_size,
            random_state=self.args.seed,
            stratify=self.y_all,
        )
        log.info("[Split] train=%d  test=%d", len(
            self.tr_text), len(self.te_text))

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------
    def vectorize(self):
        log.info("[TF-IDF] Building word model…")
        self.vect_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=self.args.max_features_word,
            sublinear_tf=True,
            stop_words="english",
            dtype=np.float32,
        )
        Xw_tr = self.vect_word.fit_transform(self.tr_text)
        Xw_te = self.vect_word.transform(self.te_text)

        if self.args.use_char:
            log.info("[TF-IDF] Adding character model…")
            self.vect_char = TfidfVectorizer(
                analyzer="char",
                ngram_range=tuple(self.args.char_ngram),
                max_features=self.args.max_features_char,
                sublinear_tf=True,
                dtype=np.float32,
            )
            Xc_tr = self.vect_char.fit_transform(self.tr_text)
            Xc_te = self.vect_char.transform(self.te_text)

            from scipy.sparse import hstack
            self.X_tr = hstack([Xw_tr, Xc_tr], format="csr")
            self.X_te = hstack([Xw_te, Xc_te], format="csr")
        else:
            self.vect_char = None
            self.X_tr, self.X_te = Xw_tr, Xw_te

        log.info("[Features] train=%s test=%s",
                 self.X_tr.shape, self.X_te.shape)

    # -----------------------------------------------------
    # EDA
    # -----------------------------------------------------
    def run_eda(self):
        if self.args.no_eda:
            return

        log.info("[EDA] generating plots…")

        # Use full dataset embeddings
        X_all = self.vect_word.transform(self.df["body_text"])
        if self.vect_char:
            from scipy.sparse import hstack
            X_all = hstack(
                [X_all, self.vect_char.transform(self.df["body_text"])],
                format="csr",
            )
        y_all = self.le.transform(self.df["label"])

        run_eda(self.df, X_all, y_all, self.classes, self.out_dir)

    # -----------------------------------------------------
    # MODEL + TRAIN
    # -----------------------------------------------------
    def build_model(self):
        dim = self.X_tr.shape[1]
        self.model = TorchLogReg(dim, self.num_classes).to(self.device)

        if self.args.class_weight == "balanced":
            binc = np.bincount(self.tr_y, minlength=self.num_classes)
            w = (len(self.tr_y) / np.clip(binc, 1, None)) / self.num_classes
            weight = torch.tensor(w, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weight)
            log.info(f"[ClassWeights] {w.tolist()}")
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def build_loaders(self):
        self.tr_loader, self.te_loader = make_loaders(
            self.X_tr, self.tr_y,
            self.X_te, self.te_y,
            self.num_classes,
            self.args,
            self.device,
        )
        log.info("[Loader] workers=%d AMP=%s",
                 self.args.num_workers, self.args.mixed_precision)

    def train(self):
        history_loss = []
        history_metric = []
        best_metric = -1
        best_state = None

        from sklearn.metrics import recall_score

        for epoch in range(1, self.args.epochs + 1):
            ep_loss = train_epoch(
                self.model,
                self.tr_loader,
                self.optimizer,
                self.criterion,
                self.device,
                self.args.mixed_precision,
                self.args.accum_steps,
            )
            history_loss.append(ep_loss)

            logits_val = predict_logits(
                self.model, self.te_loader, self.device)
            preds = logits_val.argmax(1)
            m = recall_score(self.te_y, preds, average="macro")
            history_metric.append(m)

            log.info(
                f"[Epoch {epoch}/{self.args.epochs}] loss={ep_loss:.4f} recall={m:.4f}"
            )

            if m > best_metric:
                best_metric = m
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

        # restore best
        if best_state:
            self.model.load_state_dict(best_state)

        self.history_loss = history_loss
        self.history_metric = history_metric

    # -----------------------------------------------------
    # CALIBRATION
    # -----------------------------------------------------
    def calibrate(self):
        if not self.args.calibrate:
            self.T = None
            return

        log.info("[Calibration] Temperature scaling…")
        scaler = TemperatureScaler().to(self.device)
        optT = optim.LBFGS(scaler.parameters(), lr=0.5)

        Xs, Ys = [], []
        for xb, yb in self.te_loader:
            Xs.append(xb.to(self.device))
            Ys.append(yb.to(self.device))
        Xval = torch.cat(Xs, 0)
        Yval = torch.cat(Ys, 0)

        ce = nn.CrossEntropyLoss()

        def closure():
            optT.zero_grad()
            logits = self.model(Xval)
            loss = ce(scaler(logits), Yval)
            loss.backward()
            return loss

        optT.step(closure)
        self.T = float(torch.exp(scaler.logT).cpu().item())
        log.info(f"[Calibration] T={self.T:.4f}")

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------
    def evaluate(self):
        logits = predict_logits(self.model, self.te_loader, self.device)
        if self.T:
            logits = apply_temperature(logits, self.T)

        self.probs = torch.softmax(torch.tensor(logits), 1).numpy()
        self.pred = self.probs.argmax(1)

        from sklearn.metrics import classification_report, confusion_matrix
        self.report = classification_report(
            self.te_y, self.pred, target_names=self.classes, output_dict=True
        )
        self.cm = confusion_matrix(self.te_y, self.pred).tolist()

    # -----------------------------------------------------
    # PLOTS
    # -----------------------------------------------------
    def plots(self):
        plot_confusion_matrix(self.te_y, self.pred, self.classes, self.out_dir)
        plot_training_curves(
            self.history_loss, self.history_metric, self.out_dir)
        self.pr_paths, self.roc_paths = plot_multiclass_pr_roc(
            self.te_y, self.probs, self.classes, self.out_dir
        )

    # -----------------------------------------------------
    # GPU PROFILING
    # -----------------------------------------------------
    def profile(self):
        if self.args.profile:
            run_gpu_benchmark(self.model, self.te_loader,
                              self.device, self.out_dir)

    # -----------------------------------------------------
    # SAVE ARTIFACTS
    # -----------------------------------------------------
    def save(self):
        log.info("[Save] Artifacts…")
        joblib.dump(self.vect_word, self.out_dir / "vectorizer_word.joblib")
        if self.vect_char:
            joblib.dump(self.vect_char, self.out_dir /
                        "vectorizer_char.joblib")
        joblib.dump(self.le, self.out_dir / "label_encoder.joblib")

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "input_dim": self.X_tr.shape[1],
                "classes": self.classes,
            },
            self.out_path,
        )

        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "device": str(self.device),
            "classes": self.classes,
            "class_distribution": self.df["label"].value_counts().to_dict(),
            "metrics": self.report,
            "cm": self.cm,
            "temperature": self.T,
            "sha256": self.sha,
            "plots": {
                "confusion_matrix": str(self.out_dir / "confusion_matrix.png"),
                "pr": self.pr_paths,
                "roc": self.roc_paths,
            },
        }

        with open(self.out_path.with_suffix(".metrics.json"), "w") as f:
            json.dump(metadata, f, indent=2)


# =========================================================
# CLI
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max-features-word", type=int, default=30000)
    ap.add_argument("--use-char", action="store_true")
    ap.add_argument("--char-ngram", nargs=2, type=int, default=[3, 5])
    ap.add_argument("--max-features-char", type=int, default=20000)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--accum-steps", type=int, default=1)

    ap.add_argument("--mixed-precision", action="store_true", default=True)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--class-weight",
                    choices=["none", "balanced"], default="balanced")
    ap.add_argument("--binary", action="store_true")

    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-eda", action="store_true")

    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--weighted-sampler", action="store_true", default=False)

    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        pipeline = TrainerPipeline(args)
        pipeline.load_data()
        pipeline.encode_labels()
        pipeline.split_data()
        pipeline.vectorize()
        pipeline.run_eda()
        pipeline.build_model()
        pipeline.build_loaders()
        pipeline.train()
        pipeline.calibrate()
        pipeline.evaluate()
        pipeline.plots()
        pipeline.profile()
        pipeline.save()
    except Exception as e:
        log.exception("Fatal error: %s", e)
        raise


if __name__ == "__main__":
    main()
