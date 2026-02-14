# CLI Guide

This repository exposes eight subcommands through `main.py`:

1. `train-nplm`
2. `use-nplm`
3. `train-cbow`
4. `train-cbow-negsamp`
5. `use-cbow`
6. `train-skipgram`
7. `train-skipgram-negsamp`
8. `use-skipgram`

## Feature Matrix (`use-*`)

| Capability | `use-nplm` | `use-cbow` | `use-skipgram` |
|---|---|---|---|
| Next-token prediction (`--prompt`, `--top-k`) | Yes | No | No |
| Text generation (`--generate`, `--sample`, `--temperature`) | Yes | No | No |
| Token embedding summary (`--token`, `--show-n`) | Yes | Yes | Yes |
| Nearest neighbors (`--similar`, `--top-k`) | Yes | Yes | Yes |
| Pair similarity (`--pair-sim`) | Yes | Yes | Yes |
| Analogy (`--analogy A B C`) | Yes | Yes | Yes |
| Embedding plots (`--plot-k`, `--plot-seed`, `--plot-dir`) | Yes | Yes | Yes |
| Training curves (`--plot-training-curves`) | Yes | Yes | Yes |
| Embedding export (`--export-embeddings`) | Yes | Yes | Yes |
| Exclude `<unk>` (`--no-unk`) | Yes | Yes | Yes |

## Help

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

Use custom config / overrides:

```bash
python3 main.py train-nplm --config configs/config_nplm.yaml
python3 main.py train-nplm --epochs 30 --batch-size 128 --lr 5e-4
```

## `use-nplm`

Language-model operations:

```bash
python3 main.py use-nplm --prompt "the market is" --top-k 10 --device cpu
python3 main.py use-nplm --prompt "the market is" --generate 20 --device cpu
python3 main.py use-nplm --prompt "the market is" --generate 20 --sample --temperature 0.8 --device cpu
python3 main.py use-nplm --prompt "the market is" --no-unk --top-k 10 --device cpu
```

Embedding operations (same style as `use-cbow`):

```bash
python3 main.py use-nplm --token market --show-n 8 --device cpu
python3 main.py use-nplm --similar market --top-k 10 --no-unk --device cpu
python3 main.py use-nplm --pair-sim market economy --device cpu
python3 main.py use-nplm --analogy king man woman --top-k 10 --no-unk --device cpu
python3 main.py use-nplm --export-embeddings artifacts/nplm/exports/embeddings.pt --device cpu
```

Embedding plots:

```bash
python3 main.py use-nplm --plot-k 40 --plot-seed 7 --plot-dir artifacts/nplm/plots --device cpu
```

Plot outputs:

1. PCA scatter (first two principal components)
2. `k x k` L2 distance matrix

Training curves:

```bash
python3 main.py use-nplm --plot-training-curves --device cpu
python3 main.py use-nplm --plot-training-curves \
  --metrics-path artifacts/nplm/metrics_YYYYMMDD_HHMMSS.json \
  --curves-path artifacts/nplm/plots/nplm_curves.png \
  --device cpu
```

## `train-cbow`

Train with defaults from `configs/config_cbow.yaml`:

```bash
python3 main.py train-cbow
```

Use custom config / overrides:

```bash
python3 main.py train-cbow --config configs/config_cbow.yaml
python3 main.py train-cbow --epochs 30 --batch-size 128 --lr 5e-4
```

## `train-cbow-negsamp`

Train with defaults from `configs/config_cbow_negsamp.yaml`:

```bash
python3 main.py train-cbow-negsamp
```

Use custom config / overrides:

```bash
python3 main.py train-cbow-negsamp --config configs/config_cbow_negsamp.yaml
python3 main.py train-cbow-negsamp --num-negatives 15 --negative-sampling-power 0.75
```

## `use-cbow`

Embedding operations:

```bash
python3 main.py use-cbow --token market --show-n 8 --device cpu
python3 main.py use-cbow --similar market --top-k 10 --no-unk --device cpu
python3 main.py use-cbow --pair-sim market economy --device cpu
python3 main.py use-cbow --analogy king man woman --top-k 10 --no-unk --device cpu
python3 main.py use-cbow --export-embeddings artifacts/cbow/exports/embeddings.pt --device cpu
```

Embedding plots:

```bash
python3 main.py use-cbow --plot-k 40 --plot-seed 7 --plot-dir artifacts/cbow/plots --device cpu
```

Plot outputs:

1. PCA scatter (first two principal components)
2. `k x k` L2 distance matrix

Training curves:

```bash
python3 main.py use-cbow --plot-training-curves --device cpu
python3 main.py use-cbow --plot-training-curves \
  --metrics-path artifacts/cbow/metrics_YYYYMMDD_HHMMSS.json \
  --curves-path artifacts/cbow/plots/cbow_curves.png \
  --device cpu
```

Default plot destination:

1. `cbow` checkpoint -> `artifacts/cbow/plots`
2. `cbow_negsamp` checkpoint -> `artifacts/cbow_negsamp/plots`

## `train-skipgram`

Train with defaults from `configs/config_skipgram.yaml`:

```bash
python3 main.py train-skipgram
```

Use custom config / overrides:

```bash
python3 main.py train-skipgram --config configs/config_skipgram.yaml
python3 main.py train-skipgram --epochs 30 --batch-size 128 --lr 5e-4
```

## `train-skipgram-negsamp`

Train with defaults from `configs/config_skipgram_negsamp.yaml`:

```bash
python3 main.py train-skipgram-negsamp
```

Use custom config / overrides:

```bash
python3 main.py train-skipgram-negsamp --config configs/config_skipgram_negsamp.yaml
python3 main.py train-skipgram-negsamp --num-negatives 15 --negative-sampling-power 0.75
```

## `use-skipgram`

Works with checkpoints from both:

1. `train-skipgram`
2. `train-skipgram-negsamp`

Embedding operations:

```bash
python3 main.py use-skipgram --token market --show-n 8 --device cpu
python3 main.py use-skipgram --similar market --top-k 10 --no-unk --device cpu
python3 main.py use-skipgram --pair-sim market economy --device cpu
python3 main.py use-skipgram --analogy king man woman --top-k 10 --no-unk --device cpu
python3 main.py use-skipgram --export-embeddings artifacts/skipgram/exports/embeddings.pt --device cpu
```

Embedding plots:

```bash
python3 main.py use-skipgram --plot-k 40 --plot-seed 7 --device cpu
```

Training curves:

```bash
python3 main.py use-skipgram --plot-training-curves --device cpu
```

Default plot destination:

1. `skipgram` checkpoint -> `artifacts/skipgram/plots`
2. `skipgram_negsamp` checkpoint -> `artifacts/skipgram_negsamp/plots`

## Shared Notes

1. If `--checkpoint` is omitted, latest `best_model_*.pt` under `--artifacts-dir` is used.
2. `--no-unk` excludes `<unk>` for similarity/analogy and plot sampling.
3. Plot filenames are deterministic for checkpoint + `k` and will overwrite same-name files.
4. Training supports datasets `ptb` and `text-files` with `--train-file/--valid-file/--test-file` for `text-files`.
5. CLI overrides YAML defaults.
6. `--plot-training-curves` reads `epoch_history` from metrics JSON generated by updated trainers.
7. `use-skipgram` auto-detects checkpoint type and picks default plot directory accordingly.
8. `use-cbow` auto-detects checkpoint type and picks default plot directory accordingly.

## Logs and Artifacts

Defaults:

1. Logs directory: `logs/`
2. NPLM artifacts: `artifacts/nplm/`
3. CBOW artifacts: `artifacts/cbow/`
4. CBOW negative-sampling artifacts: `artifacts/cbow_negsamp/`
5. Skip-gram artifacts: `artifacts/skipgram/`
6. Skip-gram negative-sampling artifacts: `artifacts/skipgram_negsamp/`

Timestamping behavior:

1. `best_model.pt` becomes `best_model_<timestamp>.pt`
2. `metrics.json` becomes `metrics_<timestamp>.json`
3. Auto log name is `run_<timestamp>.log`
4. Metrics JSON now includes per-epoch history (`train_loss`, `train_ppl`, `valid_loss`, `valid_ppl`).
