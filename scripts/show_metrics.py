import json, pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import precision_recall_curve, average_precision_score

# Load metrics
p = pathlib.Path("models/tfidf_lr_torch.metrics.json")
m = json.loads(p.read_text(encoding="utf-8"))

classes = m["classes"]
report = m["report"]

print("\n=== Model Evaluation Metrics ===")
print("Classes:", ", ".join(classes))
print()

print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 46)

for c in classes:
    if c in report:
        s = report[c]
        print(f"{c:<12} {s['precision']:.3f}      {s['recall']:.3f}      {s['f1-score']:.3f}")

print("-" * 46)
print(f"{'Macro Avg':<12} {report['macro avg']['precision']:.3f}      {report['macro avg']['recall']:.3f}      {report['macro avg']['f1-score']:.3f}")
print(f"{'Weighted Avg':<12} {report['weighted avg']['precision']:.3f}      {report['weighted avg']['recall']:.3f}      {report['weighted avg']['f1-score']:.3f}")
print("\nAccuracy:", m["accuracy"])

# --- Confusion Matrix ---
if "confusion_matrix" in m:
    cm = np.array(m["confusion_matrix"])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# --- Precision/Recall/F1 per class ---
precisions = [report[c]["precision"] for c in classes]
recalls = [report[c]["recall"] for c in classes]
f1s = [report[c]["f1-score"] for c in classes]

x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(8,5))
plt.bar(x - width, precisions, width, label="Precision")
plt.bar(x, recalls, width, label="Recall")
plt.bar(x + width, f1s, width, label="F1-Score")

plt.xticks(x, classes)
plt.ylim(0,1.05)
plt.ylabel("Score")
plt.title("Per-Class Precision, Recall, F1")
plt.legend()
plt.tight_layout()
plt.show()

# --- Precision-Recall Curves (if available) ---
if "pr_curves" in m:   # optional if you stored them
    for label, data in m["pr_curves"].items():
        y_true = np.array(data["y_true"])
        y_scores = np.array(data["y_scores"])
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        plt.figure(figsize=(6,5))
        plt.step(recall, precision, where="post", label=f"AP={ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve: {label}")
        plt.legend()
        plt.tight_layout()
        plt.show()
else:
    print("\n(No PR curve data found in JSON — if you want, we can add probability outputs during training so these can be plotted automatically.)")
