"""
plotting_eda.py
High-quality dataset visualization functions for PhishBlocker.
Generates EDA graphs used for academic reporting and ML analysis.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set_theme(style="whitegrid")


# ------------------------------------------------------
# SAVE
# ------------------------------------------------------
def _save(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


# ------------------------------------------------------
# BASIC DISTRIBUTIONS
# ------------------------------------------------------
def plot_class_distribution(df: pd.DataFrame, out: Path, title_suffix=""):
    counts = df["label"].value_counts().sort_index()

    # Create properly named dataframe
    counts_df = pd.DataFrame({
        "label": counts.index,
        "count": counts.values
    })

    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=counts_df,
        x="label",
        y="count",
        hue="label",
        legend=False,
        palette=["#4caf50", "#ff9800", "#e53935"],  # safe / spam / phishing
    )

    # display numbers over bars
    for i, row in counts_df.iterrows():
        plt.text(
            i,
            row["count"] + row["count"] * 0.01,
            str(row["count"]),
            ha="center",
            fontsize=10,
        )

    plt.xlabel("Class")
    plt.ylabel("Email Count")
    plt.title(f"Class Distribution {title_suffix}")
    _save(out / "class_distribution.png")


def plot_text_length(df: pd.DataFrame, out: Path):
    lens = df["body_text"].astype(str).str.len()

    plt.figure(figsize=(6, 4))
    sns.histplot(lens, bins=80, color="#4e79a7")
    plt.xlabel("Text Length (Characters)")
    plt.ylabel("Frequency")
    plt.title("Email Text Length Distribution")
    plt.grid(True, linestyle="--", alpha=0.4)
    _save(out / "text_length_hist.png")


# ------------------------------------------------------
# PCA VARIANCE
# ------------------------------------------------------
def plot_pca_variance(X_csr, out: Path):
    if X_csr.shape[1] < 10:
        return

    sample = min(3500, X_csr.shape[0])
    X = X_csr[:sample].toarray()
    pca = PCA(n_components=20)
    pca.fit(X)

    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.title("PCA Explained Variance")
    plt.xlabel("Components")
    plt.ylabel("Cumulative Variance")
    _save(out / "pca_variance.png")


# ------------------------------------------------------
# T-SNE CLUSTERS
# ------------------------------------------------------
def plot_tsne_clusters(X_csr, y, classes, out: Path):
    n = min(1500, X_csr.shape[0])
    if n < 100:
        return

    idx = np.random.choice(X_csr.shape[0], n, replace=False)
    X_sample = X_csr[idx].toarray()
    y_sample = y[idx]

    emb = TSNE(
        n_components=2,
        learning_rate="auto",
        perplexity=35,
        init="pca",
        random_state=42,
        max_iter=1000,
    ).fit_transform(X_sample)

    plt.figure(figsize=(7, 6))
    palette = ["#4caf50", "#ff9800", "#e53935"]  # safe, spam, phishing

    for i, cls in enumerate(classes):
        pts = emb[y_sample == i]
        plt.scatter(pts[:, 0], pts[:, 1], s=14,
                    alpha=0.7, label=cls, c=palette[i])

    plt.legend()
    plt.title("t-SNE Clustering in TF-IDF Feature Space")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    _save(out / "tsne_clusters.png")


# ------------------------------------------------------
# TOP NGRAMS
# ------------------------------------------------------
def simple_clean(text: str):
    # Avoid messy imports — inline stable cleaner
    text = str(text).lower()
    text = text.replace("\n", " ").replace("\r", " ")
    for ch in [".", ",", ":", ";", "(", ")", "[", "]", "!", "?", "/", "\\"]:
        text = text.replace(ch, " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def extract_ngrams(texts, n, top_k=25):
    counts = {}
    for t in texts:
        toks = t.split()
        for i in range(len(toks) - n + 1):
            gram = " ".join(toks[i:i+n])
            counts[gram] = counts.get(gram, 0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]


def plot_top_ngrams(df: pd.DataFrame, out: Path, top_k=20):
    for cls in df["label"].unique():
        texts = df[df["label"] == cls]["body_text"].astype(str)
        texts = texts.apply(simple_clean).tolist()

        uni = extract_ngrams(texts, 1, top_k)
        bi = extract_ngrams(texts, 2, top_k)

        # Unigrams
        plt.figure(figsize=(8, 5))
        keys = [x[0] for x in uni]
        vals = [x[1] for x in uni]
        sns.barplot(x=vals, y=keys, palette="Blues_r")
        plt.title(f"Top Unigrams — {cls}")
        plt.xlabel("Frequency")
        _save(out / f"top_unigrams_{cls}.png")

        # Bigrams
        plt.figure(figsize=(8, 5))
        keys_b = [x[0] for x in bi]
        vals_b = [x[1] for x in bi]
        sns.barplot(x=vals_b, y=keys_b, palette="Greens_r")
        plt.title(f"Top Bigrams — {cls}")
        plt.xlabel("Frequency")
        _save(out / f"top_bigrams_{cls}.png")


# ------------------------------------------------------
# MASTER RUN
# ------------------------------------------------------
def run_eda(df, X_csr, y, classes, out_dir: Path):
    eda = out_dir / "eda"
    eda.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(df, eda)
    plot_text_length(df, eda)
    plot_pca_variance(X_csr, eda)
    plot_tsne_clusters(X_csr, y, classes, eda)
    plot_top_ngrams(df, eda)

    print(f"[EDA] All plots saved ➜ {eda}")
