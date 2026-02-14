# cool-embeddings

`cool-embeddings` is a learning-first repository for embedding models.

Current implemented families:

1. Bengio-style Neural Probabilistic Language Model (NPLM)
2. Continuous Bag-of-Words (CBOW, full softmax and negative-sampling variants)
3. Skip-gram (full softmax and negative-sampling variants)

## Quickstart

### 1) Install

Python requirement in `pyproject.toml`: `>=3.13`.

```bash
uv sync
uv pip install torch
```

### 2) Download PTB

```bash
python3 scripts/download_PTB.py
```

### 3) Train

```bash
python3 main.py train-nplm
python3 main.py train-cbow
python3 main.py train-cbow-negsamp
python3 main.py train-skipgram
python3 main.py train-skipgram-negsamp
```

### 4) Use trained models

```bash
python3 main.py use-nplm --prompt "the market is" --top-k 10 --device cpu
python3 main.py use-cbow --similar market --top-k 10 --device cpu
python3 main.py use-skipgram --similar market --top-k 10 --device cpu
python3 main.py use-nplm --plot-training-curves --device cpu
python3 main.py use-cbow --plot-training-curves --device cpu
python3 main.py use-skipgram --plot-training-curves --device cpu
```

## CLI Help

```bash
uv run python main.py train-nplm --help
uv run python main.py use-nplm --help
uv run python main.py train-cbow --help
uv run python main.py train-cbow-negsamp --help
uv run python main.py use-cbow --help
uv run python main.py train-skipgram --help
uv run python main.py train-skipgram-negsamp --help
uv run python main.py use-skipgram --help
```

## Docs

1. Detailed CLI usage and examples: [`docs/cli.md`](docs/cli.md)
2. Project direction and roadmap: [`docs/roadmap.md`](docs/roadmap.md)

## Repository Layout

```text
.
├── main.py
├── configs/
│   ├── config_nplm.yaml
│   ├── config_skipgram.yaml
│   ├── config_cbow.yaml
│   ├── config_cbow_negsamp.yaml
│   └── config_skipgram_negsamp.yaml
├── scripts/
│   ├── download_PTB.py
│   ├── use_nplm.py
│   ├── use_cbow.py
│   └── use_skipgram.py
├── src/
│   ├── models/
│   │   ├── nplm.py
│   │   ├── cbow.py
│   │   └── skipgram.py
│   ├── train/
│   │   ├── train_nplm.py
│   │   ├── train_cbow.py
│   │   ├── train_cbow_negsamp.py
│   │   ├── train_skipgram.py
│   │   └── train_skipgram_negsamp.py
│   └── helper/
│       ├── nplm_cli.py
│       ├── cbow_cli.py
│       ├── cbow_negsamp_cli.py
│       ├── skipgram_cli.py
│       └── skipgram_negsamp_cli.py
├── docs/
│   ├── cli.md
│   └── roadmap.md
├── data/        # local datasets (gitignored)
├── logs/        # run logs (gitignored)
└── artifacts/   # checkpoints + metrics (gitignored)
```

---

Questions/comments: shekharp.erau@gmail.com
