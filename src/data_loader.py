"""
Data Loaders
============

PyTorch ``Dataset`` / ``DataLoader`` wiring for the two training tasks in
this project:

1. Word2Vec (CBOW & Skip-gram) pair datasets.
2. Sequence-to-sequence (translation) parallel-sentence datasets, with a
   padding ``collate_fn`` so variable-length sentences can be batched.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .tokenization import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, Vocab, simple_tokenizer


# --------------------------------------------------------------------------- #
# Word2Vec datasets
# --------------------------------------------------------------------------- #
class CBOWDataset(Dataset):
    """Wraps ``(context, target)`` index pairs for CBOW training."""

    def __init__(self, pairs: Sequence[Tuple[List[int], int]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


def cbow_collate_fn(batch, device: torch.device = torch.device("cpu")):
    """Flatten variable-size contexts for ``nn.EmbeddingBag``."""

    target_list, context_list, offsets = [], [], [0]
    for context, target in batch:
        target_list.append(target)
        context_tensor = torch.tensor(context, dtype=torch.int64)
        context_list.append(context_tensor)
        offsets.append(context_tensor.size(0))

    target_tensor = torch.tensor(target_list, dtype=torch.int64)
    offsets_tensor = torch.tensor(offsets[:-1]).cumsum(dim=0)
    context_tensor = torch.cat(context_list)
    return (
        context_tensor.to(device),
        offsets_tensor.to(device),
        target_tensor.to(device),
    )


class SkipGramDataset(Dataset):
    """Wraps ``(target, context_word)`` index pairs for Skip-gram training."""

    def __init__(self, pairs: Sequence[Tuple[int, int]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


def skipgram_collate_fn(batch, device: torch.device = torch.device("cpu")):
    target_list = torch.tensor([t for t, _ in batch], dtype=torch.int64)
    context_list = torch.tensor([c for _, c in batch], dtype=torch.int64)
    return target_list.to(device), context_list.to(device)


def make_word2vec_dataloader(
    dataset: Dataset, batch_size: int, collate_fn, device: torch.device, shuffle: bool = True
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, device=device),
    )


# --------------------------------------------------------------------------- #
# Sequence-to-sequence datasets
# --------------------------------------------------------------------------- #
class TranslationDataset(Dataset):
    """A small parallel-sentence dataset for the Seq2Seq translation demo.

    Each example is stored as ``<bos> ... tokens ... <eos>`` index
    sequences so the decoder learns to both start and terminate
    generation correctly.
    """

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        src_vocab: Vocab,
        trg_vocab: Vocab,
        tokenizer=simple_tokenizer,
    ) -> None:
        self.examples = []
        for src_sentence, trg_sentence in pairs:
            src_ids = [src_vocab[BOS_TOKEN]] + src_vocab.encode(tokenizer(src_sentence)) + [
                src_vocab[EOS_TOKEN]
            ]
            trg_ids = [trg_vocab[BOS_TOKEN]] + trg_vocab.encode(tokenizer(trg_sentence)) + [
                trg_vocab[EOS_TOKEN]
            ]
            self.examples.append((torch.tensor(src_ids), torch.tensor(trg_ids)))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def make_seq2seq_collate_fn(src_vocab: Vocab, trg_vocab: Vocab):
    """Return a ``collate_fn`` that pads a batch to equal length.

    Sequences are shaped ``[seq_len, batch_size]`` to match the
    ``batch_first=False`` convention used by :mod:`src.seq2seq_model`.
    """

    src_pad_idx = src_vocab[PAD_TOKEN]
    trg_pad_idx = trg_vocab[PAD_TOKEN]

    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_padded = pad_sequence(list(src_batch), padding_value=src_pad_idx)
        trg_padded = pad_sequence(list(trg_batch), padding_value=trg_pad_idx)
        return src_padded, trg_padded

    return collate_fn


def make_seq2seq_dataloader(
    dataset: Dataset, batch_size: int, collate_fn, shuffle: bool = True
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
