# CLI Guide

This repository exposes two subcommands through `main.py`:

1. `train-nplm`
2. `use-nplm`

## Help

```bash
uv run python main.py train-nplm --help
uv run python main.py use-nplm --help
```

## Dataset Setup (PTB)

```bash
python3 scripts/download_PTB.py
```

This writes:

1. `data/ptb/ptb.train.txt`
2. `data/ptb/ptb.valid.txt`
3. `data/ptb/ptb.test.txt`

## `train-nplm`

Train with defaults from `configs/config_nplm.yaml`:

```bash
python3 main.py train-nplm
```

Override config from CLI:

```bash
python3 main.py train-nplm --epochs 30 --batch-size 128 --lr 5e-4
```

Early stopping and regularization:

```bash
python3 main.py train-nplm \
  --weight-decay 1e-4 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.0
```

Device selection:

```bash
python3 main.py train-nplm --device cpu
python3 main.py train-nplm --device cuda
```

Use custom config:

```bash
python3 main.py train-nplm --config configs/config_nplm.yaml
```

Use `text-files` provider:

```bash
python3 main.py train-nplm \
  --dataset text-files \
  --train-file path/to/train.txt \
  --valid-file path/to/valid.txt \
  --test-file path/to/test.txt
```

Notes:

1. Trainer probes CUDA usability and falls back to CPU when `--device auto` is incompatible.
2. CLI overrides YAML defaults.

## `use-nplm`

Basic next-token prediction:

```bash
python3 main.py use-nplm --prompt "the market is" --top-k 10 --device cpu
```

Use a specific checkpoint:

```bash
python3 main.py use-nplm \
  --checkpoint artifacts/nplm/best_model_20260212_152242.pt \
  --prompt "the market is" \
  --top-k 10 \
  --device cpu
```

Generate continuation (greedy):

```bash
python3 main.py use-nplm --prompt "the market is" --generate 20 --device cpu
```

Generate with sampling:

```bash
python3 main.py use-nplm \
  --prompt "the market is" \
  --generate 20 \
  --sample \
  --temperature 0.8 \
  --device cpu
```

Disallow `<unk>` in top-k output and generation:

```bash
python3 main.py use-nplm \
  --prompt "the market is" \
  --generate 20 \
  --sample \
  --temperature 0.8 \
  --no-unk \
  --device cpu
```

Create embedding plots (`k` random tokens):

```bash
python3 main.py use-nplm \
  --prompt "the market is" \
  --plot-k 40 \
  --plot-seed 7 \
  --plot-dir artifacts/nplm/plots \
  --device cpu
```

Plot outputs:

1. PCA scatter (first two principal components)
2. `k x k` L2 distance matrix

Plot options:

1. `--plot-k`: number of random vocabulary tokens (`0` disables plots)
2. `--plot-seed`: seed for reproducible token sampling
3. `--plot-dir`: output directory for plot images

Notes:

1. If `--checkpoint` is omitted, latest `best_model_*.pt` under `artifacts/` is used.
2. Prompt tokenization is whitespace-based (`str.split()`).
3. Plot filenames are deterministic for checkpoint + `k` and will overwrite same-name files.

## Logs and Artifacts

Defaults:

1. Logs directory: `logs/`
2. Checkpoints directory: `artifacts/`
3. Metrics JSON directory: `artifacts/`

Set custom log file:

```bash
python3 main.py train-nplm --log-file exp1.log
```

Timestamping behavior:

1. `best_model.pt` becomes `best_model_<timestamp>.pt`
2. `metrics.json` becomes `metrics_<timestamp>.json`
3. Auto log name is `run_<timestamp>.log`

Early stopping:

1. Training may stop before max epochs if validation loss plateaus.
2. Default patience is `3` epochs.
3. Use `--early-stopping-patience 0` to disable.
