"""
Misc Utilities
===============

Small helpers for checkpointing and visualising embeddings that don't
belong in any single module above.
"""

from __future__ import annotations

import base64
import io
import os

import torch

from .tokenization import Vocab


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, map_location="cpu") -> torch.nn.Module:
    model.load_state_dict(torch.load(path, map_location=map_location))
    model.eval()
    return model


def embeddings_to_tsne_base64(embedding_matrix: torch.Tensor, vocab: Vocab, max_words: int = 60) -> str:
    """Project embeddings to 2-D with t-SNE and return a base64 PNG for web display."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    matrix = embedding_matrix[: min(max_words, embedding_matrix.shape[0])].numpy()
    words = vocab.get_itos()[: matrix.shape[0]]

    perplexity = max(5, min(30, matrix.shape[0] - 1))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
    coords = tsne.fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[:, 0], coords[:, 1], s=18, color="#6366f1")
    for i, word in enumerate(words):
        ax.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.8)
    ax.set_title("Word Embeddings (t-SNE projection)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
