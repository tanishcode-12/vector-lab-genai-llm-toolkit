#!/usr/bin/env python3
"""
app.py
======

Flask backend that serves an interactive web UI for the two trained
models:

* ``/embeddings``  -- type a word, see its nearest neighbours + a
  t-SNE plot of the embedding space.
* ``/translate``   -- type a German sentence, get an English
  translation from the Seq2Seq model, plus its BLEU score against a
  reference (if provided).

Models are trained lazily on first request (and cached in memory) so
the app can be started immediately with ``python app.py`` without a
separate training step.
"""

from __future__ import annotations

import os

import torch
from flask import Flask, jsonify, render_template, request

from src.config import (
    CHECKPOINT_DIR,
    DEFAULT_LANGUAGE_PAIR,
    EMBED_CFG,
    LANGUAGE_PAIRS,
    SEQ2SEQ_CFG,
    TOY_CORPUS_PATH,
)
from src.data_loader import (
    CBOWDataset,
    TranslationDataset,
    cbow_collate_fn,
    make_seq2seq_collate_fn,
    make_seq2seq_dataloader,
    make_word2vec_dataloader,
)
from src.embeddings import CBOW, build_cbow_pairs, get_embedding_matrix
from src.evaluate import bleu_score, most_similar_words
from src.seq2seq_model import build_seq2seq_model
from src.tokenization import EOS_TOKEN, PAD_TOKEN, build_vocab, simple_tokenizer
from src.train import train_embedding_model, train_seq2seq_model
from src.utils import embeddings_to_tsne_base64, save_checkpoint

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In-memory model cache -- populated on first use of each endpoint.
# `translators` is keyed by language-pair code (e.g. "fr-en") so each
# language trains and caches independently.
_state = {
    "embed_model": None,
    "embed_vocab": None,
    "embed_matrix": None,
    "translators": {},
}


# --------------------------------------------------------------------------- #
# Lazy model loading / training
# --------------------------------------------------------------------------- #
def get_embedding_model():
    if _state["embed_model"] is not None:
        return _state["embed_model"], _state["embed_vocab"], _state["embed_matrix"]

    with open(TOY_CORPUS_PATH, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    vocab = build_vocab(lines, specials=["<unk>"])
    tokens = simple_tokenizer(" ".join(lines))
    token_ids = vocab.encode(tokens)

    pairs = build_cbow_pairs(token_ids, EMBED_CFG.context_size)
    loader = make_word2vec_dataloader(CBOWDataset(
        pairs), EMBED_CFG.batch_size, cbow_collate_fn, DEVICE)

    model = CBOW(len(vocab), EMBED_CFG.embedding_dim).to(DEVICE)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "cbow_web.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        train_embedding_model(
            model, loader, num_epochs=EMBED_CFG.num_epochs, learning_rate=EMBED_CFG.learning_rate)
        save_checkpoint(model, checkpoint_path)

    matrix = get_embedding_matrix(model)

    _state.update(
        {"embed_model": model, "embed_vocab": vocab, "embed_matrix": matrix})
    return model, vocab, matrix


def get_translation_model(lang_pair: str = DEFAULT_LANGUAGE_PAIR):
    if lang_pair not in LANGUAGE_PAIRS:
        raise ValueError(
            f"Unknown language pair '{lang_pair}'. Choose one of {list(LANGUAGE_PAIRS)}.")

    cached = _state["translators"].get(lang_pair)
    if cached is not None:
        return cached["model"], cached["src_vocab"], cached["trg_vocab"], cached["valid_pairs"]

    info = LANGUAGE_PAIRS[lang_pair]
    pairs = []
    with open(info["file"], encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            parts = [part.strip() for part in line.strip().split("\t")]
            if len(parts) < 2:
                continue
            if info["src_lang"] == "en":
                src, trg = parts[0], parts[1]
            else:
                src, trg = parts[1], parts[0]
            pairs.append((src, trg))

    src_vocab = build_vocab([p[0] for p in pairs])
    trg_vocab = build_vocab([p[1] for p in pairs])

    split = int(len(pairs) * 0.85)
    train_pairs, valid_pairs = pairs[:split], pairs[split:]

    train_ds = TranslationDataset(train_pairs, src_vocab, trg_vocab)
    valid_ds = TranslationDataset(valid_pairs, src_vocab, trg_vocab)
    collate_fn = make_seq2seq_collate_fn(src_vocab, trg_vocab)

    train_loader = make_seq2seq_dataloader(
        train_ds, SEQ2SEQ_CFG.batch_size, collate_fn)
    valid_loader = make_seq2seq_dataloader(
        valid_ds, SEQ2SEQ_CFG.batch_size, collate_fn, shuffle=False)

    model = build_seq2seq_model(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        emb_dim=SEQ2SEQ_CFG.embedding_dim,
        hid_dim=SEQ2SEQ_CFG.hidden_dim,
        n_layers=SEQ2SEQ_CFG.num_layers,
        dropout=SEQ2SEQ_CFG.dropout,
        device=DEVICE,
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, f"seq2seq_{lang_pair}_web.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        train_seq2seq_model(
            model,
            train_loader,
            valid_loader,
            num_epochs=SEQ2SEQ_CFG.num_epochs,
            learning_rate=SEQ2SEQ_CFG.learning_rate,
            clip=SEQ2SEQ_CFG.clip,
            pad_idx=trg_vocab[PAD_TOKEN],
            device=DEVICE,
        )
        save_checkpoint(model, checkpoint_path)

    _state["translators"][lang_pair] = {
        "model": model,
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab,
        "valid_pairs": valid_pairs,
    }
    return model, src_vocab, trg_vocab, valid_pairs


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/embeddings")
def embeddings_page():
    return render_template("embeddings.html")


@app.route("/translate")
def translate_page():
    language_pairs = []
    for code, info in LANGUAGE_PAIRS.items():
        language_pairs.append(
            {
                "code": code,
                "label": info["name"],
                "source_label": f"{info['src_lang'].upper()} sentence",
                "placeholder": f"e.g. {info['example_source']}",
            }
        )
    return render_template(
        "translate.html",
        language_pairs=language_pairs,
        default_pair=DEFAULT_LANGUAGE_PAIR,
    )


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.route("/api/train-embeddings", methods=["POST"])
def api_train_embeddings():
    get_embedding_model()
    return jsonify({"status": "ready", "vocab_size": len(_state["embed_vocab"])})


@app.route("/api/similar-words", methods=["POST"])
def api_similar_words():
    data = request.get_json(force=True)
    word = (data.get("word") or "").strip().lower()

    model, vocab, matrix = get_embedding_model()
    if word not in vocab.stoi:
        return jsonify({"error": f"'{word}' is not in the vocabulary.", "vocab_sample": vocab.get_itos()[4:24]})

    neighbours = most_similar_words(word, matrix, vocab, top_k=8)
    return jsonify({"word": word, "neighbours": neighbours})


@app.route("/api/embedding-plot", methods=["GET"])
def api_embedding_plot():
    model, vocab, matrix = get_embedding_model()
    image_b64 = embeddings_to_tsne_base64(matrix, vocab, max_words=50)
    return jsonify({"image": image_b64})


@app.route("/api/vocab", methods=["GET"])
def api_vocab():
    _, vocab, _ = get_embedding_model()
    return jsonify({"words": vocab.get_itos()[4:]})  # skip special tokens


@app.route("/api/train-translator", methods=["POST"])
def api_train_translator():
    data = request.get_json(silent=True) or {}
    lang_pair = (data.get("lang_pair") or DEFAULT_LANGUAGE_PAIR).strip()
    _, src_vocab, trg_vocab, valid_pairs = get_translation_model(lang_pair)
    return jsonify(
        {
            "status": "ready",
            "lang_pair": lang_pair,
            "src_vocab_size": len(src_vocab),
            "trg_vocab_size": len(trg_vocab),
            "examples": [{"src": src, "trg": trg} for src, trg in valid_pairs],
        }
    )


@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    reference = (data.get("reference") or "").strip()
    lang_pair = (data.get("lang_pair") or DEFAULT_LANGUAGE_PAIR).strip()

    if not text:
        return jsonify({"error": "Please enter a sentence to translate."}), 400

    model, src_vocab, trg_vocab, _ = get_translation_model(lang_pair)

    src_ids = torch.tensor(
        [src_vocab["<bos>"]] + src_vocab.encode(simple_tokenizer(text)) + [src_vocab[EOS_TOKEN]])
    pred_ids = model.translate(
        src_ids, trg_vocab["<bos>"], trg_vocab[EOS_TOKEN])
    translation = " ".join(trg_vocab.decode(pred_ids))

    result = {"input": text, "translation": translation, "lang_pair": lang_pair}
    if reference:
        result["bleu"] = round(bleu_score(translation, [reference]), 4)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
