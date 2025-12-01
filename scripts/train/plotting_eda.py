"""
plotting_eda.py
High-quality dataset visualization functions for PhishBlocker.
Generates EDA graphs used for project reporting and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
from scripts.utils.text import clean_text

sns.set_theme(style="whitegrid")


def _save(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


# ------------------------------------------------------
# BASIC STATS
# ------------------------------------------------------
def plot_class_distribution(df: pd.DataFrame, out: Path):
    counts = df["label"].value_counts().sort_index()

    plt.figure(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, palette="pastel")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    _save(out / "class_distribution.png")


def plot_text_length(df: pd.DataFrame, out: Path):
    lens = df["body_text"].astype(str).str.len()

    plt.figure(figsize=(7, 4))
    sns.histplot(lens, bins=60, color="#4e79a7")
    plt.xlabel("Text Length (chars)")
    plt.ylabel("Frequency")
    plt.title("Email Text Length Distribution")
    _save(out / "text_length_hist.png")


# ------------------------------------------------------
# PCA VARIANCE
# ------------------------------------------------------
def plot_pca_variance(X_csr, out: Path):
    if X_csr.shape[1] < 10:
        return

    pca = PCA(n_components=20)
    X = X_csr[:3000].toarray()
    pca.fit(X)

    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.title("PCA Explained Variance")
    plt.xlabel("Components")
    plt.ylabel("Cumulative Variance")
    _save(out / "pca_variance.png")


# ------------------------------------------------------
# TSNE
# ------------------------------------------------------
def plot_tsne_clusters(X_csr, y, classes, out: Path):
    n = min(2500, X_csr.shape[0])
    if n < 100:
        return

    idx = np.random.choice(X_csr.shape[0], n, replace=False)
    X_sample = X_csr[idx].toarray()
    y_sample = y[idx]

    emb = TSNE(
        n_components=2,
        learning_rate=200,
        perplexity=35,
        init="pca",
        random_state=42,
        n_iter=1000,
    ).fit_transform(X_sample)

    plt.figure(figsize=(6, 5))
    for i, cls in enumerate(classes):
        mask = y_sample == i
        plt.scatter(emb[mask, 0], emb[mask, 1], s=10, label=cls)

    plt.legend()
    plt.title("t-SNE Clustering (TF-IDF)")
    _save(out / "tsne_clusters.png")


# ------------------------------------------------------
# NGRAMS (UNIGRAM / BIGRAM)
# ------------------------------------------------------
def plot_top_ngrams(df: pd.DataFrame, out: Path, top_k=20):

    def get_ngrams(texts, n):
        counts = {}
        for t in texts:
            toks = t.split()
            for i in range(len(toks) - n + 1):
                gram = " ".join(toks[i:i+n])
                counts[gram] = counts.get(gram, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

    for cls in df["label"].unique():
        texts = df[df["label"] == cls]["body_text"].astype(
            str).apply(clean_text).tolist()

        uni = get_ngrams(texts, 1)
        bi = get_ngrams(texts, 2)

        # Unigrams
        plt.figure(figsize=(7, 5))
        u_keys = [x[0] for x in uni]
        u_vals = [x[1] for x in uni]
        sns.barplot(y=u_keys, x=u_vals, palette="Blues_r")
        plt.title(f"Top Unigrams — {cls}")
        _save(out / f"top_unigrams_{cls}.png")

        # Bigrams
        plt.figure(figsize=(7, 5))
        b_keys = [x[0] for x in bi]
        b_vals = [x[1] for bi in uni]
        sns.barplot(y=b_keys, x=b_vals, palette="Greens_r")
        plt.title(f"Top Bigrams — {cls}")
        _save(out / f"top_bigrams_{cls}.png")


# ------------------------------------------------------
# MASTER FUNCTION
# ------------------------------------------------------
def run_eda(df, X_csr, y, classes, out_dir: Path):
    eda_dir = out_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(df, eda_dir)
    plot_text_length(df, eda_dir)
    plot_pca_variance(X_csr, eda_dir)
    plot_tsne_clusters(X_csr, y, classes, eda_dir)
    plot_top_ngrams(df, eda_dir)

    print(f"[EDA] All EDA plots saved to: {eda_dir}")
