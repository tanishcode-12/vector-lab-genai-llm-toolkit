#!/usr/bin/env python3
"""
run.py
======

Command-line entry point that trains both models end to end and prints
a short demo of what each one learned. This is the pure-Python
equivalent of running the original notebooks top to bottom.

Usage
-----
    python run.py embeddings              # train CBOW + Skip-gram on the toy corpus
    python run.py translate               # train Seq2Seq for every language pair
    python run.py translate --lang fr-en  # train Seq2Seq for a single language pair
    python run.py all                     # train everything (default)

Available --lang values: de-en, fr-en, es-en, it-en, all
"""

from __future__ import annotations

import argparse
import os

import torch

from src.config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    DEFAULT_LANGUAGE_PAIR,
    EMBED_CFG,
    LANGUAGE_PAIRS,
    SEQ2SEQ_CFG,
    TOY_CORPUS_PATH,
)
from src.data_loader import (
    CBOWDataset,
    SkipGramDataset,
    TranslationDataset,
    cbow_collate_fn,
    make_seq2seq_collate_fn,
    make_seq2seq_dataloader,
    make_word2vec_dataloader,
    skipgram_collate_fn,
)
from src.embeddings import (
    CBOW,
    SkipGram,
    build_cbow_pairs,
    build_skipgram_pairs,
    get_embedding_matrix,
)
from src.evaluate import bleu_score, most_similar_words, perplexity_from_loss
from src.seq2seq_model import build_seq2seq_model
from src.tokenization import EOS_TOKEN, PAD_TOKEN, build_vocab, simple_tokenizer
from src.train import train_embedding_model, train_seq2seq_model
from src.utils import save_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_embeddings_demo() -> None:
    print("\n" + "=" * 70)
    print(" WORD2VEC: TRAINING CBOW AND SKIP-GRAM ON THE TOY CORPUS")
    print("=" * 70)

    with open(TOY_CORPUS_PATH, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    vocab = build_vocab(lines, specials=["<unk>"])
    tokens = simple_tokenizer(" ".join(lines))
    token_ids = vocab.encode(tokens)
    print(f"Vocabulary size: {len(vocab)} | Total tokens: {len(token_ids)}")

    # ---- CBOW ----------------------------------------------------------- #
    print("\n[CBOW] building training pairs + training...")
    cbow_pairs = build_cbow_pairs(token_ids, EMBED_CFG.context_size)
    cbow_loader = make_word2vec_dataloader(
        CBOWDataset(cbow_pairs), EMBED_CFG.batch_size, cbow_collate_fn, DEVICE
    )
    cbow_model = CBOW(len(vocab), EMBED_CFG.embedding_dim).to(DEVICE)
    train_embedding_model(cbow_model, cbow_loader, EMBED_CFG.num_epochs, EMBED_CFG.learning_rate)
    save_checkpoint(cbow_model, os.path.join(CHECKPOINT_DIR, "cbow.pt"))

    # ---- Skip-gram -------------------------------------------------------- #
    print("\n[Skip-gram] building training pairs + training...")
    skipgram_pairs = build_skipgram_pairs(token_ids, EMBED_CFG.context_size)
    skipgram_loader = make_word2vec_dataloader(
        SkipGramDataset(skipgram_pairs), EMBED_CFG.batch_size, skipgram_collate_fn, DEVICE
    )
    skipgram_model = SkipGram(len(vocab), EMBED_CFG.embedding_dim).to(DEVICE)
    train_embedding_model(
        skipgram_model, skipgram_loader, EMBED_CFG.num_epochs, EMBED_CFG.learning_rate
    )
    save_checkpoint(skipgram_model, os.path.join(CHECKPOINT_DIR, "skipgram.pt"))

    # ---- demo: nearest neighbours ----------------------------------------- #
    print("\nNearest neighbours (Skip-gram embeddings):")
    matrix = get_embedding_matrix(skipgram_model)
    for query in ["small", "big", "beautiful", "team"]:
        neighbours = most_similar_words(query, matrix, vocab, top_k=5)
        words = ", ".join(n["word"] for n in neighbours) if neighbours else "(not in vocab)"
        print(f"  {query:>12s}  ->  {words}")


def run_translation_demo(lang_pair: str = DEFAULT_LANGUAGE_PAIR) -> None:
    info = LANGUAGE_PAIRS[lang_pair]
    print("\n" + "=" * 70)
    print(f" SEQ2SEQ: TRAINING A TOY {info['src_lang'].upper()} -> ENGLISH TRANSLATOR")
    print("=" * 70)

    pairs = []
    with open(info["file"], encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            en, src = line.strip().split("\t")
            pairs.append((src, en))  # source = foreign language, target = English

    src_vocab = build_vocab([p[0] for p in pairs])
    trg_vocab = build_vocab([p[1] for p in pairs])
    print(f"Source (de) vocab: {len(src_vocab)} | Target (en) vocab: {len(trg_vocab)}")

    split = int(len(pairs) * 0.85)
    train_pairs, valid_pairs = pairs[:split], pairs[split:]

    train_ds = TranslationDataset(train_pairs, src_vocab, trg_vocab)
    valid_ds = TranslationDataset(valid_pairs, src_vocab, trg_vocab)
    collate_fn = make_seq2seq_collate_fn(src_vocab, trg_vocab)

    train_loader = make_seq2seq_dataloader(train_ds, SEQ2SEQ_CFG.batch_size, collate_fn)
    valid_loader = make_seq2seq_dataloader(valid_ds, SEQ2SEQ_CFG.batch_size, collate_fn, shuffle=False)

    model = build_seq2seq_model(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        emb_dim=SEQ2SEQ_CFG.embedding_dim,
        hid_dim=SEQ2SEQ_CFG.hidden_dim,
        n_layers=SEQ2SEQ_CFG.num_layers,
        dropout=SEQ2SEQ_CFG.dropout,
        device=DEVICE,
    )
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_losses, valid_losses = train_seq2seq_model(
        model,
        train_loader,
        valid_loader,
        num_epochs=SEQ2SEQ_CFG.num_epochs,
        learning_rate=SEQ2SEQ_CFG.learning_rate,
        clip=SEQ2SEQ_CFG.clip,
        pad_idx=trg_vocab[PAD_TOKEN],
        device=DEVICE,
    )
    save_checkpoint(model, os.path.join(CHECKPOINT_DIR, f"seq2seq_{lang_pair}.pt"))

    print(f"\nFinal validation perplexity: {perplexity_from_loss(valid_losses[-1]):.2f}")

    print("\nSample translations:")
    for src_sentence, en_sentence in valid_pairs[:5]:
        src_ids = torch.tensor(
            [src_vocab["<bos>"]] + src_vocab.encode(simple_tokenizer(src_sentence)) + [src_vocab[EOS_TOKEN]]
        )
        pred_ids = model.translate(src_ids, trg_vocab["<bos>"], trg_vocab[EOS_TOKEN])
        prediction = " ".join(trg_vocab.decode(pred_ids))
        bleu = bleu_score(prediction, [en_sentence])
        print(f"  SRC ({info['src_lang']}): {src_sentence}")
        print(f"  REF: {en_sentence}")
        print(f"  HYP: {prediction}   (BLEU={bleu:.2f})\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Generative AI / LLM toolkit models.")
    parser.add_argument(
        "task",
        nargs="?",
        default="all",
        choices=["embeddings", "translate", "all"],
        help="Which pipeline to run (default: all)",
    )
    parser.add_argument(
        "--lang",
        default="all",
        choices=list(LANGUAGE_PAIRS.keys()) + ["all"],
        help="Language pair(s) to train for the translate task (default: all)",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    if args.task in ("embeddings", "all"):
        run_embeddings_demo()
    if args.task in ("translate", "all"):
        lang_pairs = list(LANGUAGE_PAIRS.keys()) if args.lang == "all" else [args.lang]
        for lang_pair in lang_pairs:
            run_translation_demo(lang_pair)

    print("\nDone. Checkpoints saved to:", CHECKPOINT_DIR)
    print("Run `python app.py` to explore the models in the web UI.")


if __name__ == "__main__":
    main()