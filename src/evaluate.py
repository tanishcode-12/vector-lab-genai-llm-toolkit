"""
Evaluation Utilities
=====================

* :func:`most_similar_words` -- cosine-similarity nearest-neighbour
  search over a trained embedding matrix.
* :func:`perplexity_from_loss` -- converts a cross-entropy loss into
  perplexity, the standard intrinsic language-model metric.
* :func:`bleu_score` -- corpus/sentence BLEU for evaluating translation
  quality (falls back to a small pure-Python implementation if
  ``nltk`` is not installed).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch

from .tokenization import Vocab


# --------------------------------------------------------------------------- #
# Embedding similarity
# --------------------------------------------------------------------------- #
def most_similar_words(
    word: str, embedding_matrix: torch.Tensor, vocab: Vocab, top_k: int = 5
) -> List[Dict[str, float]]:
    """Return the ``top_k`` words whose embedding is closest to ``word``.

    Uses cosine similarity, which is scale-invariant and the standard
    choice for comparing Word2Vec-style embeddings.
    """

    if word not in vocab.stoi:
        return []

    idx = vocab[word]
    target_vec = embedding_matrix[idx]
    target_norm = target_vec.norm() + 1e-8

    similarities = (embedding_matrix @ target_vec) / (
        embedding_matrix.norm(dim=1) * target_norm + 1e-8
    )

    top_indices = torch.argsort(similarities, descending=True).tolist()
    results = []
    for i in top_indices:
        candidate = vocab.itos[i]
        if candidate == word:
            continue
        results.append({"word": candidate, "similarity": round(similarities[i].item(), 4)})
        if len(results) == top_k:
            break
    return results


# --------------------------------------------------------------------------- #
# Language-model metrics
# --------------------------------------------------------------------------- #
def perplexity_from_loss(loss: float) -> float:
    """Perplexity = exp(cross-entropy loss). Lower is better."""

    return math.exp(loss) if loss < 20 else float("inf")


# --------------------------------------------------------------------------- #
# BLEU
# --------------------------------------------------------------------------- #
def _ngram_counts(tokens: Sequence[str], n: int) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def _simple_bleu(hypothesis: Sequence[str], references: List[Sequence[str]], max_n: int = 4) -> float:
    """A compact, dependency-free BLEU implementation (used as a fallback)."""

    if len(hypothesis) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        hyp_counts = _ngram_counts(hypothesis, n)
        if not hyp_counts:
            precisions.append(0.0)
            continue

        max_ref_counts: Dict[tuple, int] = {}
        for ref in references:
            ref_counts = _ngram_counts(ref, n)
            for ngram, cnt in ref_counts.items():
                max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), cnt)

        clipped = sum(min(cnt, max_ref_counts.get(ngram, 0)) for ngram, cnt in hyp_counts.items())
        total = sum(hyp_counts.values())
        precisions.append(clipped / total if total > 0 else 0.0)

    if min(precisions) == 0.0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    ref_len = min(references, key=lambda r: abs(len(r) - len(hypothesis)))
    hyp_len, closest_ref_len = len(hypothesis), len(ref_len)
    brevity_penalty = 1.0 if hyp_len > closest_ref_len else math.exp(1 - closest_ref_len / max(hyp_len, 1))

    return geo_mean * brevity_penalty


def bleu_score(hypothesis: str, references: Sequence[str]) -> float:
    """Compute the BLEU score of ``hypothesis`` against one or more ``references``.

    Both ``hypothesis`` and ``references`` are plain, whitespace
    tokenizable strings (e.g. ``"the cat sat"``).
    """

    hyp_tokens = hypothesis.split()
    ref_tokens_list = [ref.split() for ref in references]

    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        smoothing = SmoothingFunction().method1
        return sentence_bleu(ref_tokens_list, hyp_tokens, smoothing_function=smoothing)
    except ImportError:
        return _simple_bleu(hyp_tokens, ref_tokens_list)
