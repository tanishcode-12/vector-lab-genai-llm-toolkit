"""
Central configuration for the project.

Keeping every hyperparameter in one place makes the pipeline easy to
tune from a single file instead of hunting through every module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
TOY_CORPUS_PATH = os.path.join(DATA_DIR, "toy_corpus.txt")
PARALLEL_CORPUS_PATH = os.path.join(DATA_DIR, "parallel_corpus.tsv")
DEFAULT_LANGUAGE_PAIR = "de-en"
LANGUAGE_PAIRS = {
    "de-en": {
        "file": PARALLEL_CORPUS_PATH,
        "src_lang": "de",
        "trg_lang": "en",
        "name": "German → English",
        "example_source": "der mann geht",
        "example_target": "the man is walking",
    },
    "en-de": {
        "file": PARALLEL_CORPUS_PATH,
        "src_lang": "en",
        "trg_lang": "de",
        "name": "English → German",
        "example_source": "the man is walking",
        "example_target": "der mann geht",
    },
    "es-en": {
        "file": os.path.join(DATA_DIR, "parallel_es_en.tsv"),
        "src_lang": "es",
        "trg_lang": "en",
        "name": "Spanish → English",
        "example_source": "el hombre camina",
        "example_target": "the man is walking",
    },
    "en-es": {
        "file": os.path.join(DATA_DIR, "parallel_es_en.tsv"),
        "src_lang": "en",
        "trg_lang": "es",
        "name": "English → Spanish",
        "example_source": "the man is walking",
        "example_target": "el hombre camina",
    },
    "fr-en": {
        "file": os.path.join(DATA_DIR, "parallel_fr_en.tsv"),
        "src_lang": "fr",
        "trg_lang": "en",
        "name": "French → English",
        "example_source": "l'homme marche",
        "example_target": "the man is walking",
    },
    "en-fr": {
        "file": os.path.join(DATA_DIR, "parallel_fr_en.tsv"),
        "src_lang": "en",
        "trg_lang": "fr",
        "name": "English → French",
        "example_source": "the man is walking",
        "example_target": "l'homme marche",
    },
    "it-en": {
        "file": os.path.join(DATA_DIR, "parallel_it_en.tsv"),
        "src_lang": "it",
        "trg_lang": "en",
        "name": "Italian → English",
        "example_source": "l'uomo cammina",
        "example_target": "the man is walking",
    },
    "en-it": {
        "file": os.path.join(DATA_DIR, "parallel_it_en.tsv"),
        "src_lang": "en",
        "trg_lang": "it",
        "name": "English → Italian",
        "example_source": "the man is walking",
        "example_target": "l'uomo cammina",
    },
}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


@dataclass
class EmbeddingConfig:
    """Hyperparameters for the Word2Vec (CBOW / Skip-gram) models."""

    context_size: int = 2
    embedding_dim: int = 32
    batch_size: int = 64
    learning_rate: float = 5.0
    num_epochs: int = 150


@dataclass
class Seq2SeqConfig:
    """Hyperparameters for the LSTM encoder-decoder translation model."""

    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    batch_size: int = 8
    learning_rate: float = 1e-3
    num_epochs: int = 120
    teacher_forcing_ratio: float = 0.5
    clip: float = 1.0


EMBED_CFG = EmbeddingConfig()
SEQ2SEQ_CFG = Seq2SeqConfig()
