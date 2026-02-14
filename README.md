# cool-embeddings

`cool-embeddings` is a learning-first repository for embedding models.

Current implemented families:

1. Bengio-style Neural Probabilistic Language Model (NPLM)
2. Continuous Bag-of-Words (CBOW)

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
```

### 4) Use trained models

```bash
python3 main.py use-nplm --prompt "the market is" --top-k 10 --device cpu
python3 main.py use-cbow --similar market --top-k 10 --device cpu
```

## CLI Help

```bash
uv run python main.py train-nplm --help
uv run python main.py use-nplm --help
uv run python main.py train-cbow --help
uv run python main.py use-cbow --help
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
│   └── config_cbow.yaml
├── scripts/
│   ├── download_PTB.py
│   ├── use_nplm.py
│   └── use_cbow.py
├── src/
│   ├── models/
│   │   ├── nplm.py
│   │   └── cbow.py
│   ├── train/
│   │   ├── train_nplm.py
│   │   └── train_cbow.py
│   └── helper/
│       ├── nplm_cli.py
│       └── cbow_cli.py
├── docs/
│   ├── cli.md
│   └── roadmap.md
├── data/        # local datasets (gitignored)
├── logs/        # run logs (gitignored)
└── artifacts/   # checkpoints + metrics (gitignored)
```

---

Questions/comments: shekharp.erau@gmail.com
