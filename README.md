# cool-embeddings

`cool-embeddings` is a learning-first, implementation-driven repository for embedding models.

The repo starts from early neural language models (current baseline: Bengio 2003 NPLM) and is designed to grow step-by-step toward modern embedding systems.

## Goal

Build a single codebase that tracks the evolution of embeddings:

1. Classic feedforward neural language models
2. Word embeddings (Word2Vec/GloVe/FastText-style workflows)
3. Contextual sequence models (RNN/LSTM/Transformer encoders)
4. Contrastive and sentence embeddings
5. Modern large-scale embedding approaches

This means the code is intentionally modular and easy to extend rather than optimized only for one model.

## Current Status

Implemented today:

1. Neural Probabilistic Language Model (NPLM) in PyTorch
2. PTB dataset download script
3. Config-driven training with CLI overrides
4. Per-run logs
5. Timestamped artifacts (no overwrite between runs)

## Repository Layout

```text
.
├── main.py                      # project CLI entrypoint
├── configs/
│   └── config_nplm.yaml         # default training config
├── scripts/
│   └── download_PTB.py          # PTB downloader
├── src/
│   ├── models/
│   │   └── nplm.py              # Bengio-style NPLM model
│   ├── train/
│   │   └── train_nplm.py        # training orchestration
│   └── helper/
│       ├── nplm_cli.py          # config + CLI parsing/overrides
│       ├── nplm_data.py         # dataset providers, vocab, dataloaders
│       └── nplm_logging.py      # run logging/tee utilities
├── data/                        # local datasets (ignored by git)
├── logs/                        # per-run terminal logs (ignored)
└── artifacts/                   # checkpoints + metrics (ignored)
```

## Environment Setup

Python requirement in `pyproject.toml` is currently `>=3.13`.

Install dependencies:

```bash
pip install pyyaml torch
```

Or with `uv`:

```bash
uv sync
uv pip install torch
```

Note: `torch` is intentionally installed explicitly depending on your CPU/GPU setup.

## Dataset: PTB

Download PTB splits:

```bash
python3 scripts/download_PTB.py
```

This writes:

1. `data/ptb/ptb.train.txt`
2. `data/ptb/ptb.valid.txt`
3. `data/ptb/ptb.test.txt`

## Training

All training runs through `main.py`.

Run with defaults from `configs/config_nplm.yaml`:

```bash
python3 main.py train-nplm
```

Override config from CLI:

```bash
python3 main.py train-nplm --epochs 30 --batch-size 128 --lr 5e-4
```

Early stopping and regularization controls:

```bash
python3 main.py train-nplm \
  --weight-decay 1e-4 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.0
```

Choose device explicitly:

```bash
python3 main.py train-nplm --device cpu
python3 main.py train-nplm --device cuda
```

The trainer does a CUDA usability probe. If `device=auto` and CUDA is incompatible, it falls back to CPU with a warning.

## Config System

Default config file: `configs/config_nplm.yaml`

Sections:

1. `data`
2. `model`
3. `optimization`
4. `runtime`
5. `output`
6. `logging`

CLI always overrides YAML defaults.

Use a custom config:

```bash
python3 main.py train-nplm --config configs/config_nplm.yaml
```

## Logs and Artifacts

Each run records full terminal output to a log file.

Defaults:

1. Logs directory: `logs/`
2. Checkpoints directory: `artifacts/`
3. Metrics JSON directory: `artifacts/`

You can set a custom log filename:

```bash
python3 main.py train-nplm --log-file exp1.log
```

Timestamping behavior:

1. `best_model.pt` becomes `best_model_<timestamp>.pt`
2. `metrics.json` becomes `metrics_<timestamp>.json`
3. Auto log name is `run_<timestamp>.log`

This prevents overwriting outputs across runs.

Early-stopping behavior:

1. Training can stop before `epochs` if validation loss does not improve.
2. Default patience is `3` epochs.
3. Set `--early-stopping-patience 0` to disable early stopping.

## Data Providers (Extensible)

Current dataset options:

1. `ptb`
2. `text-files`

Example for arbitrary text files:

```bash
python3 main.py train-nplm \
  --dataset text-files \
  --train-file path/to/train.txt \
  --valid-file path/to/valid.txt \
  --test-file path/to/test.txt
```

## Model Summary (Current Baseline)

`src/models/nplm.py` implements the Bengio 2003 feedforward NPLM:

1. Context token embeddings
2. Concatenation of context embeddings
3. Hidden layer with `tanh`
4. Vocabulary projection
5. Optional direct context-to-output connection

## Recommended PTB Training Defaults

For this baseline model, a good starting point is:

1. `epochs: 20-30`
2. monitor validation perplexity
3. `weight_decay: 1e-4`
4. keep best checkpoint by validation loss

Current config defaults (`configs/config_nplm.yaml`) include:

1. `weight_decay: 1e-4`
2. `early_stopping_patience: 3`
3. `early_stopping_min_delta: 0.0`

## Known Notes

1. This repo is intentionally at an early stage.
2. Structure is being prepared for many embedding families, not just NPLM.
3. Additional benchmark datasets and model families will be added incrementally.

## Roadmap

Planned additions (high-level):

1. Word2Vec (CBOW/Skip-gram) training pipeline
2. GloVe-style co-occurrence factorization
3. FastText-style subword embeddings
4. LSTM/Transformer language-model embeddings
5. Sentence embedding objectives (contrastive/triplet/dual-encoder)
6. Evaluation suite (word similarity, retrieval, transfer tasks)
7. Unified experiment tracking and reproducibility tooling

## Contributing

If you add a new model family:

1. Keep `main.py` as the single entrypoint
2. Keep configs in `configs/`
3. Keep heavy outputs in `artifacts/` and logs in `logs/`
4. Add docs and CLI examples to this README

---

This repository is meant to be a living progression from foundational embedding ideas to current best-practice systems.

In case of questions/comments/suggestions please drop me an email at: shekharp.erau@gmail.com
