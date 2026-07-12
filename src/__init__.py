"""
genai-llm-toolkit
==================

A from-scratch PyTorch implementation of the core building blocks behind
modern generative language models:

    * text tokenization & vocabulary construction
    * batching / padding data loaders
    * Word2Vec embeddings (CBOW and Skip-gram)
    * an LSTM Encoder-Decoder (Sequence-to-Sequence) translation model
    * evaluation utilities (perplexity, BLEU, nearest-neighbour search)

The package is intentionally dependency-light (pure PyTorch) so every
component can be read, trained and inspected end to end.
"""

__version__ = "1.0.0"
