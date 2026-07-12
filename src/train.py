"""
Training Loops
===============

Generic, well-logged training loops shared by both tasks in the
project. Kept deliberately simple (no external experiment trackers) so
the mechanics of the optimisation step are easy to follow.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .embeddings import CBOW


def train_embedding_model(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    log_every: int = 20,
) -> List[float]:
    """Train a CBOW or Skip-gram model, returning the per-epoch loss history."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

    is_cbow = isinstance(model, CBOW)
    epoch_losses: List[float] = []

    model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()

            if is_cbow:
                context, offsets, target = batch
                predicted = model(context, offsets)
            else:
                target, context = batch
                predicted = model(target)

            loss = criterion(predicted, context if not is_cbow else target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / max(len(dataloader), 1)
        epoch_losses.append(avg_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == num_epochs:
            print(f"  epoch {epoch:4d}/{num_epochs} | loss = {avg_loss:.4f}")

    return epoch_losses


def train_seq2seq_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    device: torch.device,
) -> float:
    """Run a single training epoch over the Seq2Seq model."""

    model.train()
    epoch_loss = 0.0

    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg)  # [trg_len, batch, vocab]
        output_dim = output.shape[-1]

        output = output[1:].reshape(-1, output_dim)
        trg_flat = trg[1:].reshape(-1)

        loss = criterion(output, trg_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_seq2seq_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """Compute average loss on a validation set (teacher forcing disabled)."""

    model.eval()
    epoch_loss = 0.0

    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg, teacher_forcing_ratio=0.0)
        output_dim = output.shape[-1]

        output = output[1:].reshape(-1, output_dim)
        trg_flat = trg[1:].reshape(-1)

        loss = criterion(output, trg_flat)
        epoch_loss += loss.item()

    return epoch_loss / max(len(dataloader), 1)


def train_seq2seq_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    clip: float,
    pad_idx: int,
    device: torch.device,
    log_every: int = 10,
):
    """Full training loop with validation, returning loss histories."""

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    train_losses, valid_losses = [], []
    for epoch in range(1, num_epochs + 1):
        train_loss = train_seq2seq_epoch(model, train_loader, optimizer, criterion, clip, device)
        valid_loss = evaluate_seq2seq_epoch(model, valid_loader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == num_epochs:
            print(
                f"  epoch {epoch:4d}/{num_epochs} | "
                f"train_loss = {train_loss:.4f} | valid_loss = {valid_loss:.4f}"
            )

    return train_losses, valid_losses
