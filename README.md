<div align="center">

# 🧠✨ Vector Lab — Generative AI & LLM Toolkit ✨🧠

### A from-scratch, dependency-light PyTorch playground for the building blocks of modern LLMs 🚀

![banner](https://raw.githubusercontent.com/tanishcode-12/vector-lab-genai-llm-toolkit/main/assets/banner.gif)
<!-- 👆 replace this with your own banner GIF (drop it in an /assets folder). A quick screen-recording of the t-SNE plot animating looks 🔥 -->

![status](https://img.shields.io/badge/status-active-6EE7D8?style=for-the-badge)
![python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![pytorch](https://img.shields.io/badge/pytorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![flask](https://img.shields.io/badge/flask-web%20app-000000?style=for-the-badge&logo=flask&logoColor=white)
![license](https://img.shields.io/badge/license-MIT-purple?style=for-the-badge)
![made_with_love](https://img.shields.io/badge/made%20with-%E2%9D%A4%EF%B8%8F%20%26%20caffeine-red?style=for-the-badge)

![building blocks meme](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)
<!-- 😂 classic "it's actually simple" meme — swap for whatever meme energy fits your vibe -->

</div>

---

## 🤯 Nothing here is a black box

Tokenization 🔤 → Padded batching 📦 → Word2Vec embeddings (CBOW & Skip-gram) 🧩 → LSTM Encoder–Decoder (Seq2Seq) translation 🌍 — plus a **polished web UI** 🎨 to explore it all interactively.

Every architecture is implemented in `src/` and can be read start to finish. The web app trains models **live, in-process** — zero external API calls, zero mystery. 🕵️‍♂️

> 💡 **TL;DR** — this repo is what happens when you refuse to import `transformers` and build the whole thing yourself for fun.

---

## 📚 What's inside

| Module | Concept | What it does |
|---|---|---|
| 🔤 `src/tokenization.py` | Tokenization & vocabulary | Word-level tokenizer + a `Vocab` class with `<pad>`/`<unk>`/`<bos>`/`<eos>` specials |
| 📦 `src/data_loader.py` | Batching & padding | `Dataset`/`DataLoader` wiring, including `pad_sequence`-based collation for variable-length sentences |
| 🧩 `src/embeddings.py` | Word2Vec | `CBOW` (EmbeddingBag + mean pooling) and `SkipGram` models, trained from scratch on a toy corpus |
| 🔁 `src/seq2seq_model.py` | Encoder–Decoder | An LSTM `Encoder`, `Decoder`, and `Seq2Seq` wrapper with teacher forcing and greedy decoding |
| 🏋️ `src/train.py` | Training loops | Shared, logged training loops for both tasks |
| 📊 `src/evaluate.py` | Evaluation | Cosine-similarity nearest neighbours, perplexity, and BLEU (with an NLTK-free fallback) |
| 🛠️ `src/utils.py` | Utilities | Checkpointing, t-SNE embedding visualization |

---

## 🌐 Web app

`app.py` serves three pages, each with its own vibe:

- 🏠 **Overview** — an animated vector-space visualization + a walkthrough of the architecture
- 🔍 **Embedding Explorer** — type a word, get its nearest neighbours by cosine similarity, plus a **live t-SNE plot** ✨
- 🌍 **Translator** — a toy German → English Seq2Seq demo with BLEU scoring against a reference sentence

![embedding explorer demo](https://media.giphy.com/media/l0HlNQ03J5JxX6lva/giphy.gif)
<!-- 🎥 swap for a real screen-recording GIF of your Embedding Explorer in action -->

Models train lazily on first request and are cached in memory (and on disk in `checkpoints/`) for the rest of the session. ⚡

---

## 🗂️ Project structure

```
genai-llm-toolkit/
├── app.py                    # 🌐 Flask backend
├── run.py                    # 🖥️ CLI: train + demo both pipelines from the terminal
├── requirements.txt          # 📦 dependencies
├── src/
│   ├── config.py             # ⚙️ all hyperparameters live here
│   ├── tokenization.py       # 🔤
│   ├── data_loader.py        # 📦
│   ├── embeddings.py         # 🧩
│   ├── seq2seq_model.py      # 🔁
│   ├── train.py              # 🏋️
│   ├── evaluate.py           # 📊
│   └── utils.py              # 🛠️
├── data/
│   ├── toy_corpus.txt        # 📖 training text for CBOW / Skip-gram
│   └── parallel_corpus.tsv   # 🌍 DE-EN sentence pairs for the translator
├── templates/                # 🖼️ Jinja2 HTML (base, index, embeddings, translate)
├── static/
│   ├── css/style.css         # 💅
│   └── js/                   # vector-field.js, embeddings.js, translate.js
└── checkpoints/               # 💾 trained weights (git-ignored, generated on first run)
```

---

## 🚀 Setup

```bash
git clone https://github.com/tanishcode-12/vector-lab-genai-llm-toolkit.git
cd vector-lab-genai-llm-toolkit
python3 -m venv venv && source venv/bin/activate      # optional but recommended 🐍
pip install -r requirements.txt
```

## ▶️ Usage

### 🌐 Web app (recommended)

```bash
python app.py
```

Then open **http://localhost:5000** 🔗

Training happens automatically the first time you query the embedding explorer or the translator (roughly 15–30 seconds on CPU ⏱️) and is cached for the rest of the session.

![loading meme](https://media.giphy.com/media/3oriO0OEd9QIDdllqo/giphy.gif)
<!-- ⏳ "training in progress" meme placeholder — perfect for the 15-30s wait -->

### 🖥️ Command line

```bash
python run.py embeddings   # 🧩 train CBOW + Skip-gram, print nearest-neighbour demo
python run.py translate    # 🌍 train the Seq2Seq translator, print sample translations + BLEU
python run.py all          # 🔥 both (default)
```

---

## 🏗️ Architecture notes

### 🧩 CBOW
Predicts a target word from the *mean* of its surrounding context words — fast to train, and a good fit for frequent words:

```
context words → EmbeddingBag(mean) → ReLU → Linear → target logits
```

### 🎯 Skip-gram
Does the reverse, predicting each context word from the target — slower per epoch (more training pairs) but generally produces better representations for rare words:

```
target word → Embedding → ReLU → Linear → context logits
```

### 🔁 Seq2Seq
A classic Sutskever-style LSTM encoder–decoder. The encoder compresses the whole source sentence into a final `(hidden, cell)` state; the decoder is unrolled one token at a time, starting from `<bos>`, using either the ground-truth previous token (*teacher forcing* 👨‍🏫, during training) or its own last prediction (during inference):

```
src tokens → [Encoder LSTM] → (hidden, cell)
                                    │
                                    ▼
<bos> → [Decoder step] → tok₁ → [Decoder step] → tok₂ → ... → <eos>
```

---

## 📈 Evaluation

| Metric | What it means |
|---|---|
| 📉 **Perplexity** | `exp(cross_entropy_loss)` — the standard intrinsic language-model metric; lower is better |
| 🎯 **BLEU** | n-gram precision (up to 4-grams) with a brevity penalty, scored against one or more reference translations |
| 🧭 **Cosine similarity** | used for nearest-neighbour search over the learned embedding matrix |

---

## 📏 A note on scale

The toy corpora (`data/toy_corpus.txt`, `data/parallel_corpus.tsv`) are **intentionally tiny** 🐣 so the whole pipeline trains in well under a minute on CPU and is easy to inspect end-to-end.

Swap in a larger corpus (e.g. the Multi30k dataset 🌍) and bump up `EmbeddingConfig` / `Seq2SeqConfig` in `src/config.py` to scale this into a production-grade model — **the architecture code does not need to change.** 💪

![scale meme](https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif)
<!-- 📈 "it's over 9000" / scaling-up meme placeholder -->

---

## 🧰 Tech stack

🐍 Python &nbsp;·&nbsp; 🔥 PyTorch &nbsp;·&nbsp; 🌐 Flask &nbsp;·&nbsp; 🔢 NumPy &nbsp;·&nbsp; 🤖 scikit-learn (t-SNE) &nbsp;·&nbsp; 📝 NLTK (BLEU) &nbsp;·&nbsp; 🎨 vanilla HTML/CSS/JS (no frontend build step required)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repo
2. Create your branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🩹 Recent fixes

- ✅ Fixed t-SNE plot to refresh on each new word query instead of staying static.
- ✅ Fixed `DEFAULT_LANGUAGE_PAIR` import error in `src/config.py`.

---

## 📜 License

MIT — use freely, attribution appreciated. 🙏

---

<div align="center">

### ⭐ If this repo helped you understand LLMs a little better, drop it a star! ⭐

![thanks meme](https://media.giphy.com/media/g9582DNuQppxC/giphy.gif)

</div>