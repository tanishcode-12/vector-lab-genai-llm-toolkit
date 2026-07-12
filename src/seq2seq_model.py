"""
Sequence-to-Sequence (Encoder-Decoder) Translation Model
=========================================================

A classic LSTM encoder-decoder architecture (Sutskever et al., 2014
style) used here for a small toy German -> English translation task.

Architecture
------------
::

    src tokens -> [Encoder LSTM] -> (hidden, cell)
                                         |
                                         v
    <bos> -> [Decoder LSTM step] -> word_1 -> [Decoder LSTM step] -> word_2 -> ...

The encoder compresses the whole source sentence into a fixed-size
``(hidden, cell)`` state. The decoder is then unrolled one token at a
time, using either the *previous ground-truth token* (teacher forcing)
or its *own previous prediction* as the next input -- this is what
:pyattr:`Seq2Seq.teacher_forcing_ratio` controls.
"""

from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Compresses a source sequence into a final ``(hidden, cell)`` state."""

    def __init__(
        self, vocab_size: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # src: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # outputs are unused -- only the final hidden/cell state is passed on
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    """Generates the target sentence one token at a time."""

    def __init__(
        self, vocab_size: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.output_dim = vocab_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout if n_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, input_token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_token: [batch_size] -> [1, batch_size]
        input_token = input_token.unsqueeze(0)
        embedded = self.dropout(self.embedding(input_token))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))  # [batch_size, vocab_size]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """Wires the encoder and decoder together with (optional) teacher forcing."""

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device) -> None:
        super().__init__()
        assert encoder.hid_dim == decoder.hid_dim, "Encoder/decoder hidden dims must match"
        assert encoder.n_layers == decoder.n_layers, "Encoder/decoder layer counts must match"

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        # src: [src_len, batch_size]   trg: [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size, device=self.device)

        hidden, cell = self.encoder(src)

        # first decoder input is always the <bos> token
        input_token = trg[0, :]

        for t in range(1, trg_len):
            prediction, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[t] = prediction

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = trg[t] if teacher_force else top1

        return outputs

    @torch.no_grad()
    def translate(
        self,
        src_indices: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 30,
    ) -> list:
        """Greedy-decode a single source sequence -> list of predicted token ids."""

        self.eval()
        src_indices = src_indices.unsqueeze(1).to(self.device)  # [src_len, 1]
        hidden, cell = self.encoder(src_indices)

        input_token = torch.tensor([bos_idx], device=self.device)
        predicted_indices = []

        for _ in range(max_len):
            prediction, hidden, cell = self.decoder(input_token, hidden, cell)
            top1 = prediction.argmax(1)
            token_id = top1.item()
            if token_id == eos_idx:
                break
            predicted_indices.append(token_id)
            input_token = top1

        return predicted_indices


def build_seq2seq_model(
    src_vocab_size: int,
    trg_vocab_size: int,
    emb_dim: int,
    hid_dim: int,
    n_layers: int,
    dropout: float,
    device: torch.device,
) -> Seq2Seq:
    """Convenience factory that builds and moves a full Seq2Seq model to ``device``."""

    encoder = Encoder(src_vocab_size, emb_dim, hid_dim, n_layers, dropout)
    decoder = Decoder(trg_vocab_size, emb_dim, hid_dim, n_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)
    return model
