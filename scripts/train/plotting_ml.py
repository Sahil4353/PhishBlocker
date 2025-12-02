"""
plotting_ml.py
High-quality ML evaluation & model performance plots.
Designed for academic reports and ML deliverables.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

sns.set_theme(style="whitegrid")

# ------------------------------
# Path-safe save helper
# ------------------------------


def _save(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


# ------------------------------
# 1️⃣ Confusion Matrix
# ------------------------------
def plot_confusion_matrix(y_true, y_pred, classes, out: Path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    _save(out / "confusion_matrix.png")


# ------------------------------
# 2️⃣ Multi-class PR & ROC
# ------------------------------
def plot_multiclass_pr_roc(y_true, probs, classes, out: Path):
    pr_files = {}
    roc_files = {}

    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        prob_pos = probs[:, i]

        if y_bin.sum() == 0:
            # no positives for this class; skip to avoid roc_auc_score error
            continue

        # ---- PR curve ----
        prec, rec, _ = precision_recall_curve(y_bin, prob_pos)
        ap = average_precision_score(y_bin, prob_pos)

        plt.figure(figsize=(6, 4))
        plt.plot(rec, prec, linewidth=2)
        plt.title(f"Precision–Recall | Class: {cls} (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True, linestyle="--", alpha=0.4)
        pth = out / f"pr_{cls}.png"
        _save(pth)
        pr_files[cls] = str(pth)

        # ---- ROC curve ----
        fpr, tpr, _ = roc_curve(y_bin, prob_pos)
        auc = roc_auc_score(y_bin, prob_pos)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"ROC | Class: {cls}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True, linestyle="--", alpha=0.4)
        pth = out / f"roc_{cls}.png"
        _save(pth)
        roc_files[cls] = str(pth)

    return pr_files, roc_files


# ------------------------------
# 3️⃣ Calibration Reliability Diagram
# ------------------------------
def plot_calibration_curve(probs, y_true, out: Path, bins=10):
    """
    Reliability diagram: predicted probability vs empirical accuracy.
    """
    p = probs.max(axis=1)
    bin_edges = np.linspace(0, 1, bins + 1)
    accs = []
    confs = []

    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (p >= lo) & (p < hi)
        if mask.sum():
            confs.append(p[mask].mean())
            accs.append(
                (np.argmax(probs[mask], axis=1) == y_true[mask]).mean())
        else:
            confs.append((lo + hi) / 2)
            accs.append(0)

    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.plot(confs, accs, marker="o", linewidth=2)
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration Reliability Diagram")
    plt.grid(True, linestyle="--", alpha=0.4)
    _save(out / "calibration.png")


# ------------------------------
# 4️⃣ Epoch curves
# ------------------------------
def plot_training_curves(losses, recalls, out: Path):
    epochs = np.arange(1, len(losses) + 1)

    # Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, marker="o", linewidth=2)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    _save(out / "training_loss.png")

    # Recall metric
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, recalls, marker="o", linewidth=2, color="#2ca02c")
    plt.title("Validation Recallish")
    plt.xlabel("Epoch")
    plt.ylabel("Recallish")
    plt.grid(True, linestyle="--", alpha=0.4)
    _save(out / "training_recallish.png")
