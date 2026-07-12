"""
Tokenization & Vocabulary
=========================

A minimal, dependency-free tokenizer and vocabulary implementation.

Design notes
------------
* Tokenization here is *word-based* (lower-cased, punctuation stripped)
  which keeps the vocabulary small and easy to inspect for teaching
  purposes. Swapping in a subword tokenizer (e.g. BPE / WordPiece) only
  requires changing :func:`simple_tokenizer`.
* ``Vocab`` mirrors the interface of ``torchtext.vocab.Vocab`` closely
  enough (``__len__``, ``__getitem__``, ``get_itos``) that the rest of
  the codebase does not need to know which implementation backs it.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def simple_tokenizer(text: str) -> List[str]:
    """Lower-case word tokenizer.

    Example
    -------
    >>> simple_tokenizer("The dog runs on the ground!")
    ['the', 'dog', 'runs', 'on', 'the', 'ground']
    """

    return _TOKEN_RE.findall(text.lower())


class Vocab:
    """A simple bidirectional token <-> index mapping.

    Parameters
    ----------
    tokens:
        An iterable of already-tokenized text (a list of token lists).
    min_freq:
        Tokens occurring fewer than ``min_freq`` times are dropped and
        mapped to ``<unk>`` at lookup time.
    specials:
        Special tokens that are always placed at the front of the
        vocabulary (index 0, 1, 2, ...).
    """

    def __init__(
        self,
        tokens: Iterable[Sequence[str]],
        min_freq: int = 1,
        specials: Sequence[str] = SPECIAL_TOKENS,
    ) -> None:
        counter: Counter = Counter()
        for sentence in tokens:
            counter.update(sentence)

        self.itos: List[str] = list(specials)
        for token, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= min_freq and token not in self.itos:
                self.itos.append(token)

        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self._unk_index = self.stoi.get(UNK_TOKEN, 0)

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self._unk_index)

    def get_itos(self) -> List[str]:
        return list(self.itos)

    def encode(self, sentence: Sequence[str]) -> List[int]:
        """Convert a list of tokens into a list of vocabulary indices."""

        return [self[token] for token in sentence]

    def decode(self, indices: Sequence[int]) -> List[str]:
        """Convert a list of vocabulary indices back into tokens."""

        return [self.itos[i] if 0 <= i < len(self.itos) else UNK_TOKEN for i in indices]


def build_vocab(
    sentences: Iterable[str],
    tokenizer=simple_tokenizer,
    min_freq: int = 1,
    specials: Sequence[str] = SPECIAL_TOKENS,
) -> Vocab:
    """Build a :class:`Vocab` from raw sentence strings."""

    tokenized = [tokenizer(sentence) for sentence in sentences]
    return Vocab(tokenized, min_freq=min_freq, specials=specials)
