#!/usr/bin/env python
"""
train_baseline.py
Clean OOP orchestrator for PhishBlocker TF-IDF baseline model.

FULL 2025 REFACTOR:
- 3-way split: TRAIN / CALIBRATION / FINAL-TEST
- calibration uses ONLY calibration set (no leakage)
- evaluation uses ONLY final test set
- TF-IDF feature extraction
- fast CSR → dense loader
- AMP + AdamW + Accumulation
- temperature scaling
- full metrics + PR/ROC per class
- GPU profiling optional
"""

from __future__ import annotations

# ===============================
# STD LIB
# ===============================
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime, timezone

# ===============================
# LIBS
# ===============================
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# ===============================
# PROJECT
# ===============================
from scripts.train.data_loader import make_loaders
from scripts.train.model import TorchLogReg, TemperatureScaler
from scripts.train.trainer import train_epoch, evaluate, predict_logits
from scripts.train.plotting_eda import run_eda
from scripts.train.plotting_ml import (
    plot_confusion_matrix,
    plot_multiclass_pr_roc,
    plot_training_curves,
)
from scripts.train.profiling_gpu import run_gpu_benchmark
from scripts.train.utils_common import (
    seed_everything,
    load_concat_df,
    canonicalize_labels,
    apply_temperature,
)


log = logging.getLogger("pipeline")


# =========================================================
# PIPELINE
# =========================================================
class TrainerPipeline:
    def __init__(self, args):
        self.args = args

        seed_everything(args.seed)
        self.out_path = Path(args.out)
        self.out_dir = self.out_path.parent
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"[Device] {self.device}")

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------
    def load_data(self):
        df, sha = load_concat_df(self.args.inputs)
        df = canonicalize_labels(df)

        if self.args.binary:
            df["label"] = df["label"].map(
                {"safe": "safe", "spam": "not_safe", "phishing": "not_safe"}
            )

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

    # -----------------------------------------------------
    # three-way split
    # -----------------------------------------------------
    def split_data(self):
        Xtxt = self.df["body_text"].astype(str).values
        y = self.y_all

        # split train vs remainder
        X_train, X_rest, y_train, y_rest = train_test_split(
            Xtxt,
            y,
            test_size=self.args.val_cal_size + self.args.test_size,
            random_state=self.args.seed,
            stratify=y,
        )

        # split cal vs test
        cal_ratio = self.args.val_cal_size / \
            (self.args.val_cal_size + self.args.test_size)
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_rest,
            y_rest,
            test_size=self.args.test_size /
            (self.args.val_cal_size + self.args.test_size),
            random_state=self.args.seed,
            stratify=y_rest,
        )

        self.tr_text, self.tr_y = X_train, y_train
        self.X_cal, self.y_cal = X_cal, y_cal
        self.X_test, self.y_test = X_test, y_test

        log.info(
            "[Split] train=%d | cal=%d | test=%d",
            len(self.tr_text),
            len(self.X_cal),
            len(self.X_test),
        )

    # -----------------------------------------------------
    # FEATURE EXTRACTION
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
        Xw_cal = self.vect_word.transform(self.X_cal)
        Xw_te = self.vect_word.transform(self.X_test)

        if self.args.use_char:
            log.info("[TF-IDF] Adding char model…")
            self.vect_char = TfidfVectorizer(
                analyzer="char",
                ngram_range=tuple(self.args.char_ngram),
                max_features=self.args.max_features_char,
                sublinear_tf=True,
                dtype=np.float32,
            )
            Xc_tr = self.vect_char.fit_transform(self.tr_text)
            Xc_cal = self.vect_char.transform(self.X_cal)
            Xc_te = self.vect_char.transform(self.X_test)

            from scipy.sparse import hstack
            self.X_tr = hstack([Xw_tr, Xc_tr], format="csr")
            self.X_cal = hstack([Xw_cal, Xc_cal], format="csr")
            self.X_test = hstack([Xw_te, Xc_te], format="csr")
        else:
            self.vect_char = None
            self.X_tr = Xw_tr
            self.X_cal = Xw_cal
            self.X_test = Xw_te

        log.info(
            f"[Features] train={self.X_tr.shape} cal={self.X_cal.shape} test={self.X_test.shape}")

    # -----------------------------------------------------
    # EDA
    # -----------------------------------------------------
    def run_eda(self):
        if self.args.no_eda:
            return

        log.info("[EDA] generating plots…")

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
    # MODEL
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
        # train loader
        self.tr_loader, self.cal_loader = make_loaders(
            self.X_tr, self.tr_y,
            self.X_cal, self.y_cal,
            self.num_classes,
            self.args,
            self.device,
        )
        # final test loader
        _, self.te_loader = make_loaders(
            self.X_tr, self.tr_y,
            self.X_test, self.y_test,
            self.num_classes,
            self.args,
            self.device,
        )

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------
    def train(self):
        log.info("[Training]")
        best_metric = -1
        best_state = None

        from sklearn.metrics import recall_score
        loss_hist = []
        recall_hist = []

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
            loss_hist.append(ep_loss)

            logits = predict_logits(self.model, self.cal_loader, self.device)
            preds = logits.argmax(1)
            m = recall_score(self.y_cal, preds, average="macro")
            recall_hist.append(m)

            log.info(
                f"[Epoch {epoch}/{self.args.epochs}] loss={ep_loss:.4f} cal_recall={m:.4f}")

            if m > best_metric:
                best_metric = m
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}

        if best_state:
            self.model.load_state_dict(best_state)

        self.history_loss = loss_hist
        self.history_metric = recall_hist

    # -----------------------------------------------------
    # CALIBRATION
    # -----------------------------------------------------
    def calibrate(self):
        if not self.args.calibrate:
            self.T = None
            return

        log.info("[Calibration] Temperature scaling on cal split…")

        logits = predict_logits(self.model, self.cal_loader, self.device)
        Yval = np.concatenate([yb.numpy() for _, yb in self.cal_loader])

        logits_t = torch.tensor(logits, device=self.device)
        y_t = torch.tensor(Yval, device=self.device)

        scaler = TemperatureScaler().to(self.device)
        optT = optim.LBFGS(scaler.parameters(), lr=0.5)
        ce = nn.CrossEntropyLoss()

        def closure():
            optT.zero_grad()
            out = scaler(logits_t)
            loss = ce(out, y_t)
            loss.backward()
            return loss

        optT.step(closure)
        self.T = float(torch.exp(scaler.logT).cpu().item())
        log.info(f"[Calibration] T={self.T:.4f}")

    # -----------------------------------------------------
    # FINAL EVAL
    # -----------------------------------------------------
    def evaluate(self):
        logits = predict_logits(self.model, self.te_loader, self.device)
        if self.T:
            logits = apply_temperature(logits, self.T)

        self.probs = torch.softmax(torch.tensor(logits), 1).numpy()
        self.pred = self.probs.argmax(1)

        acc, rep, cm = evaluate(
            self.model, self.te_loader, self.device, self.classes)
        self.final_acc, self.report, self.cm = acc, rep, cm

        log.info(f"[FinalTest] accuracy={acc:.4f}")

    # -----------------------------------------------------
    # PLOTS
    # -----------------------------------------------------
    def plots(self):
        plot_confusion_matrix(self.y_test, self.pred,
                              self.classes, self.out_dir)
        plot_training_curves(
            self.history_loss, self.history_metric, self.out_dir)
        self.pr_paths, self.roc_paths = plot_multiclass_pr_roc(
            self.y_test, self.probs, self.classes, self.out_dir
        )

    # -----------------------------------------------------
    # GPU PROFILING
    # -----------------------------------------------------
    def profile(self):
        if self.args.profile:
            run_gpu_benchmark(self.model, self.te_loader,
                              self.device, self.out_dir)

    # -----------------------------------------------------
    # SAVE
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
            "final_accuracy": self.final_acc,
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

    # Input/Output
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)

    # splits
    ap.add_argument("--val-cal-size", type=float, default=0.15,
                    help="validation/calibration split")
    ap.add_argument("--test-size", type=float, default=0.15,
                    help="final test split")
    ap.add_argument("--seed", type=int, default=42)

    # TF-IDF
    ap.add_argument("--max-features-word", type=int, default=30000)
    ap.add_argument("--use-char", action="store_true")
    ap.add_argument("--char-ngram", nargs=2, type=int, default=[3, 5])
    ap.add_argument("--max-features-char", type=int, default=20000)

    # training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--mixed-precision", action="store_true", default=True)
    ap.add_argument("--num-workers", type=int, default=2)

    # labels
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--class-weight",
                    choices=["none", "balanced"], default="balanced")
    ap.add_argument("--weighted-sampler", action="store_true", default=False)

    # calibration
    ap.add_argument("--calibrate", action="store_true")

    # profiling
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-eda", action="store_true")

    # opt
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.01)

    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        pipe = TrainerPipeline(args)
        pipe.load_data()
        pipe.encode_labels()
        pipe.split_data()
        pipe.vectorize()
        pipe.run_eda()
        pipe.build_model()
        pipe.build_loaders()
        pipe.train()
        pipe.calibrate()
        pipe.evaluate()
        pipe.plots()
        pipe.profile()
        pipe.save()
    except Exception as e:
        log.exception("Fatal error: %s", e)
        raise


if __name__ == "__main__":
    main()
