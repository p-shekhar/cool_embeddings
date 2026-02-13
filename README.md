# cool-embeddings

`cool-embeddings` is a learning-first repository for embedding models.

Current baseline: Bengio (2003)-style Neural Probabilistic Language Model (NPLM) in PyTorch.

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
```

### 4) Use trained model

```bash
python3 main.py use-nplm --prompt "the market is" --top-k 10 --device cpu
```

## CLI Help

```bash
uv run python main.py train-nplm --help
uv run python main.py use-nplm --help
```

## Docs

1. Detailed CLI usage and examples: `docs/cli.md`
2. Project direction, baseline notes, and roadmap: `docs/roadmap.md`

## Repository Layout

```text
.
├── main.py
├── configs/
│   └── config_nplm.yaml
├── scripts/
│   ├── download_PTB.py
│   └── use_nplm.py
├── src/
│   ├── models/
│   │   └── nplm.py
│   ├── train/
│   │   └── train_nplm.py
│   └── helper/
├── docs/
│   ├── cli.md
│   └── roadmap.md
├── data/        # local datasets (gitignored)
├── logs/        # run logs (gitignored)
└── artifacts/   # checkpoints + metrics (gitignored)
```

## Contributing

If you add a new model family:

1. Keep `main.py` as the single entrypoint.
2. Keep configs in `configs/`.
3. Keep heavy outputs in `artifacts/` and logs in `logs/`.
4. Update docs in `docs/`.

---

Questions/comments: shekharp.erau@gmail.com
