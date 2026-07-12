"""
Word2Vec Embeddings: CBOW & Skip-gram
======================================

From-scratch PyTorch implementations of the two classic Word2Vec
architectures:

* **CBOW** (Continuous Bag of Words) predicts a *target* word from its
  surrounding *context* words. The context is aggregated with an
  ``nn.EmbeddingBag`` (mean pooling), which is fast and works well when
  the context window is small.

* **Skip-gram** does the reverse: it predicts each *context* word given
  the *target* word, which tends to produce better embeddings for rare
  words at the cost of more training pairs.

Both models expose the trained embedding matrix so the vectors can be
inspected, visualised (t-SNE) or queried for nearest neighbours -- see
:mod:`src.evaluate`.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .tokenization import Vocab


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class CBOW(nn.Module):
    """Continuous Bag-of-Words model.

    context words --(EmbeddingBag, mean-pooled)--> hidden --(ReLU + Linear)--> vocab logits
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.linear1 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc = nn.Linear(embed_dim // 2, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def forward(self, context: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        pooled = self.embedding(context, offsets)
        hidden = torch.relu(self.linear1(pooled))
        return self.fc(hidden)


class SkipGram(nn.Module):
    """Skip-gram model.

    target word --(Embedding)--> hidden --(ReLU + Linear)--> vocab logits
    (trained on (target, context_word) pairs)
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.embeddings(target))
        return self.fc(hidden)


# --------------------------------------------------------------------------- #
# Training-pair construction
# --------------------------------------------------------------------------- #
def build_cbow_pairs(
    token_ids: Sequence[int], context_size: int
) -> List[Tuple[List[int], int]]:
    """Slide a window over ``token_ids`` producing ``(context, target)`` pairs."""

    pairs = []
    for i in range(context_size, len(token_ids) - context_size):
        context = (
            [token_ids[i - context_size + j] for j in range(context_size)]
            + [token_ids[i + j + 1] for j in range(context_size)]
        )
        pairs.append((context, token_ids[i]))
    return pairs


def build_skipgram_pairs(
    token_ids: Sequence[int], context_size: int
) -> List[Tuple[int, int]]:
    """Build ``(target, context_word)`` pairs for every word in the window."""

    pairs = []
    for i in range(context_size, len(token_ids) - context_size):
        target = token_ids[i]
        for j in range(1, context_size + 1):
            pairs.append((target, token_ids[i - j]))
            pairs.append((target, token_ids[i + j]))
    return pairs


def get_embedding_matrix(model: nn.Module) -> torch.Tensor:
    """Return the learned embedding matrix for either model type."""

    if isinstance(model, CBOW):
        return model.embedding.weight.detach().cpu()
    if isinstance(model, SkipGram):
        return model.embeddings.weight.detach().cpu()
    raise TypeError(f"Unsupported model type: {type(model)!r}")
