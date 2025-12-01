#!/usr/bin/env python
"""
train_baseline.py
Orchestrator entrypoint for PhishBlocker 3-class / binary TF-IDF models.

Responsibilities:
- CLI parsing
- loading CSV
- building TF-IDF features
- calling EDA module
- constructing loaders
- model + training
- calibration and evaluation
- saving artifacts
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

# =============== PROJECT MODULES ===================
from scripts.train.data_loader import make_loaders
from scripts.train.model import TorchLogReg, TemperatureScaler
from scripts.train.trainer import train_epoch, evaluate, predict_logits
from scripts.train.profiling_gpu import run_gpu_benchmark
from scripts.train.plotting_eda import run_eda
from scripts.train.plotting_ml import (
    plot_multiclass_pr_roc,
    plot_confusion_matrix,
    plot_training_curves
)
from scripts.train.utils_common import (
    seed_everything,
    load_concat_df,
    canonicalize_labels,
    tune_threshold,
    apply_temperature,
)

# Builtin libs
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger("train")


# ============================================================
# =========== MAIN HIGH-LEVEL ORCHESTRATION ==================
# ============================================================
def run(args: argparse.Namespace):

    # Setup
    seed_everything(args.seed)
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("[*] Loading datasets …")
    df, sha256 = load_concat_df(args.inputs)
    df = canonicalize_labels(df)

    # Optional binary collapse
    if args.binary:
        df["label"] = df["label"].map({
            "safe": "safe",
            "spam": "not_safe",
            "phishing": "not_safe"
        })

    X_text = df["body_text"].astype(str).values
    y_text = df["label"].values

    # Encode
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y_text)
    y = le.transform(y_text)
    classes = list(le.classes_)
    num_classes = len(classes)

    log.info(f"Classes: {classes}")

    # --------- Train / Test Split ------------
    from sklearn.model_selection import train_test_split
    Xtr_txt, Xte_txt, ytr, yte = train_test_split(
        X_text,
        y,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y,
    )

    # -------- Vectorization ------------
    log.info("[*] Building TF-IDF features …")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vect_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=args.max_features_word,
        sublinear_tf=True,
        stop_words="english",
        dtype=np.float32,
    )
    Xw_tr = vect_word.fit_transform(Xtr_txt)
    Xw_te = vect_word.transform(Xte_txt)

    # Char model?
    if args.use_char:
        vect_char = TfidfVectorizer(
            analyzer="char",
            ngram_range=tuple(args.char_ngram),
            max_features=args.max_features_char,
            sublinear_tf=True,
            dtype=np.float32,
        )
        Xc_tr = vect_char.fit_transform(Xtr_txt)
        Xc_te = vect_char.transform(Xte_txt)

        from scipy.sparse import hstack
        X_tr_csr = hstack([Xw_tr, Xc_tr], format="csr")
        X_te_csr = hstack([Xw_te, Xc_te], format="csr")
    else:
        vect_char = None
        X_tr_csr, X_te_csr = Xw_tr, Xw_te

    # ---------------------------------------------------------
    # -------- EDA VISUALS (skip on very large) ---------------
    # ---------------------------------------------------------
    if not args.no_eda:
        run_eda(df, X_tr_csr, ytr, classes, out_dir)

    # ---------------------------------------------------------
    # -------- DATA LOADERS -----------------------------------
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_loader, te_loader = make_loaders(
        X_tr_csr, ytr,
        X_te_csr, yte,
        num_classes,
        args,
        device,
    )
    log.info(f"Device: {device} | Mixed Precision: {args.mixed_precision}")

    # ---------------------------------------------------------
    # -------- MODEL ------------------------------------------
    # ---------------------------------------------------------
    model = TorchLogReg(X_tr_csr.shape[1], num_classes).to(device)

    # Loss weights (class imbalance)
    if args.class_weight == "balanced":
        binc = np.bincount(ytr, minlength=num_classes)
        w = (len(ytr) / np.clip(binc, 1, None)) / num_classes
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        log.info(f"Class weights: {w.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---------------------------------------------------------
    # -------- TRAIN LOOP -------------------------------------
    # ---------------------------------------------------------
    history_loss = []
    history_metric = []
    best_metric = -1
    best_state = None

    for epoch in range(1, args.epochs + 1):
        ep_loss = train_epoch(
            model,
            tr_loader,
            optimizer,
            criterion,
            device,
            args.mixed_precision,
            args.accum_steps,
        )
        history_loss.append(ep_loss)

        # macro recall
        from sklearn.metrics import recall_score
        logits_val = predict_logits(model, te_loader, device)
        preds = logits_val.argmax(1)
        m = recall_score(yte, preds, average="macro")
        history_metric.append(m)

        log.info(
            f"Epoch {epoch}/{args.epochs} | loss={ep_loss:.4f} | recall={m:.4f}")

        if m > best_metric:
            best_metric = m
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # ---------------------------------------------------------
    # -------- TEMPERATURE SCALING ----------------------------
    # ---------------------------------------------------------
    T = None
    if args.calibrate:
        scaler = TemperatureScaler().to(device)
        optT = optim.LBFGS(scaler.parameters(), lr=0.5)

        # gather val
        Xs = []
        Ys = []
        for xb, yb in te_loader:
            Xs.append(xb.to(device))
            Ys.append(yb.to(device))
        Xv = torch.cat(Xs, 0)
        Yv = torch.cat(Ys, 0)

        ce = nn.CrossEntropyLoss()

        def closure():
            optT.zero_grad()
            logits = model(Xv)
            logits_T = scaler(logits)
            loss = ce(logits_T, Yv)
            loss.backward()
            return loss

        optT.step(closure)
        T = float(torch.exp(scaler.logT).cpu().item())
        log.info(f"[Calibration] Temperature: {T:.4f}")

    # ---------------------------------------------------------
    # -------- FINAL EVAL -------------------------------------
    # ---------------------------------------------------------
    logits_final = predict_logits(model, te_loader, device)
    if T:
        logits_final = apply_temperature(logits_final, T)

    probs = torch.softmax(torch.tensor(logits_final), 1).numpy()
    pred = probs.argmax(1)

    # metrics
    from sklearn.metrics import classification_report, confusion_matrix
    rep = classification_report(
        yte, pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(yte, pred).tolist()

    # ---------------------------------------------------------
    # ---------- PLOTS ----------------------------------------
    # ---------------------------------------------------------
    plot_confusion_matrix(cm, classes, out_dir / "confusion_matrix.png")
    plot_training_curves(history_loss, history_metric, out_dir)
    pr_paths, roc_paths = plot_multiclass_pr_roc(yte, probs, classes, out_dir)

    # ---------------------------------------------------------
    # -------- SAVE ARTIFACTS ---------------------------------
    # ---------------------------------------------------------
    joblib.dump(vect_word, out_dir / "vectorizer_word.joblib")
    if vect_char:
        joblib.dump(vect_char, out_dir / "vectorizer_char.joblib")
    joblib.dump(le, out_dir / "label_encoder.joblib")

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": X_tr_csr.shape[1],
            "classes": classes,
        },
        out_path,
    )

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "classes": classes,
        "class_distribution": df["label"].value_counts().to_dict(),
        "metrics": rep,
        "cm": cm,
        "temperature": T,
        "plots": {
            "confusion_matrix": str(out_dir / "confusion_matrix.png"),
            "pr_curves": pr_paths,
            "roc_curves": roc_paths,
        },
        "sha256": sha256,
    }

    with open(out_path.with_suffix(".metrics.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("[✓] Done.")
    return 0


# ============================================================
# CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--inputs", nargs="+", required=True)
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

    ap.add_argument("--out", required=True)
    ap.add_argument("--no-eda", action="store_true")

    ap.add_argument("--log-level", default="INFO")

    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        sys.exit(run(args))
    except Exception as e:
        log.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
