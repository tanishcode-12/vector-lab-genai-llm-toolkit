# Vector Lab — Generative AI & LLM Toolkit

A from-scratch, dependency-light PyTorch implementation of the core building
blocks behind modern language models: tokenization, padded batching, Word2Vec
embeddings (CBOW & Skip-gram), and an LSTM encoder–decoder (Seq2Seq)
translation model — plus a polished web UI to explore both interactively.

Nothing here is a black box. Every architecture is implemented in `src/` and
can be read start to finish; the web app trains the models live, in-process,
with no external API calls.

![status](https://img.shields.io/badge/status-active-6EE7D8?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![pytorch](https://img.shields.io/badge/pytorch-2.x-EE4C2C?style=flat-square)

---

## What's inside

| Module | Concept | What it does |
|---|---|---|
| `src/tokenization.py` | Tokenization & vocabulary | Word-level tokenizer + a `Vocab` class with `<pad>`/`<unk>`/`<bos>`/`<eos>` specials |
| `src/data_loader.py` | Batching & padding | `Dataset`/`DataLoader` wiring, including `pad_sequence`-based collation for variable-length sentences |
| `src/embeddings.py` | Word2Vec | `CBOW` (EmbeddingBag + mean pooling) and `SkipGram` models, trained from scratch on a toy corpus |
| `src/seq2seq_model.py` | Encoder–Decoder | An LSTM `Encoder`, `Decoder`, and `Seq2Seq` wrapper with teacher forcing and greedy decoding |
| `src/train.py` | Training loops | Shared, logged training loops for both tasks |
| `src/evaluate.py` | Evaluation | Cosine-similarity nearest neighbours, perplexity, and BLEU (with an NLTK-free fallback) |
| `src/utils.py` | Utilities | Checkpointing, t-SNE embedding visualization |

## Web app

`app.py` serves three pages:

- **Overview** — an animated vector-space visualization and a walkthrough of the architecture
- **Embedding Explorer** — type a word, get its nearest neighbours by cosine similarity, plus a live t-SNE plot
- **Translator** — a toy German → English Seq2Seq demo with BLEU scoring against a reference sentence

Models train lazily on first request and are cached in memory (and on disk in
`checkpoints/`) for the rest of the session.

---

## Project structure

```
genai-llm-toolkit/
├── app.py                    # Flask backend
├── run.py                    # CLI: train + demo both pipelines from the terminal
├── requirements.txt
├── src/
│   ├── config.py             # all hyperparameters live here
│   ├── tokenization.py
│   ├── data_loader.py
│   ├── embeddings.py
│   ├── seq2seq_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── data/
│   ├── toy_corpus.txt        # training text for CBOW / Skip-gram
│   └── parallel_corpus.tsv   # DE-EN sentence pairs for the translator
├── templates/                # Jinja2 HTML (base, index, embeddings, translate)
├── static/
│   ├── css/style.css
│   └── js/                   # vector-field.js, embeddings.js, translate.js
└── checkpoints/               # trained weights (git-ignored, generated on first run)
```

---

## Setup

```bash
git clone https://github.com/yourusername/vector-lab.git
cd vector-lab
python3 -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

## Usage

### Web app (recommended)

```bash
python app.py
```

Then open **http://localhost:5000**. Training happens automatically the
first time you query the embedding explorer or the translator (roughly
15–30 seconds on CPU) and is cached for the rest of the session.

### Command line

```bash
python run.py embeddings   # train CBOW + Skip-gram, print nearest-neighbour demo
python run.py translate    # train the Seq2Seq translator, print sample translations + BLEU
python run.py all          # both (default)
```

---

## Architecture notes

**CBOW** predicts a target word from the *mean* of its surrounding context
words — fast to train, and a good fit for frequent words:

```
context words → EmbeddingBag(mean) → ReLU → Linear → target logits
```

**Skip-gram** does the reverse, predicting each context word from the
target — slower per epoch (more training pairs) but generally produces
better representations for rare words:

```
target word → Embedding → ReLU → Linear → context logits
```

**Seq2Seq** is a classic Sutskever-style LSTM encoder–decoder. The encoder
compresses the whole source sentence into a final `(hidden, cell)` state;
the decoder is unrolled one token at a time, starting from `<bos>`, using
either the ground-truth previous token (*teacher forcing*, during training)
or its own last prediction (during inference):

```
src tokens → [Encoder LSTM] → (hidden, cell)
                                    │
                                    ▼
<bos> → [Decoder step] → tok₁ → [Decoder step] → tok₂ → ... → <eos>
```

### Evaluation

- **Perplexity** = `exp(cross_entropy_loss)` — the standard intrinsic
  language-model metric; lower is better.
- **BLEU** — n-gram precision (up to 4-grams) with a brevity penalty,
  scored against one or more reference translations.
- **Cosine similarity** — used for nearest-neighbour search over the
  learned embedding matrix.

### A note on scale

The toy corpora (`data/toy_corpus.txt`, `data/parallel_corpus.tsv`) are
intentionally small so the whole pipeline trains in well under a minute on
CPU and is easy to inspect end-to-end. Swap in a larger corpus (e.g. the
Multi30k dataset) and increase `EmbeddingConfig` / `Seq2SeqConfig` in
`src/config.py` to scale this up into a production-grade model — the
architecture code does not need to change.

---

## Tech stack

Python · PyTorch · Flask · NumPy · scikit-learn (t-SNE) · NLTK (BLEU) ·
vanilla HTML/CSS/JS (no frontend build step required)

## License

MIT — use freely, attribution appreciated.
