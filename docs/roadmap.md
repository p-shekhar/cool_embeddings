# Project Direction

`cool-embeddings` is a learning-first, implementation-driven repository for embedding models.

The current baseline is a Bengio (2003)-style Neural Probabilistic Language Model (NPLM), and the codebase is structured to evolve toward broader embedding systems.

## Goal

Build one codebase that tracks the evolution of embeddings:

1. Classic feedforward neural language models
2. Word embeddings (Word2Vec/GloVe/FastText-style workflows)
3. Contextual sequence models (RNN/LSTM/Transformer encoders)
4. Contrastive and sentence embeddings
5. Modern large-scale embedding approaches

## Current Status

Implemented now:

1. NPLM in PyTorch
2. PTB downloader
3. Config-driven training with CLI overrides
4. Per-run logging
5. Timestamped artifacts to avoid overwrite

## Model Baseline

`src/models/nplm.py` implements feedforward NPLM:

1. Context token embeddings
2. Concatenated context representation
3. Hidden layer with `tanh`
4. Vocabulary projection
5. Optional direct context-to-output connection

## Suggested PTB Baseline Defaults

1. `epochs: 20-30`
2. Monitor validation perplexity
3. `weight_decay: 1e-4`
4. Keep best checkpoint by validation loss

Current defaults in `configs/config_nplm.yaml`:

1. `weight_decay: 1e-4`
2. `early_stopping_patience: 3`
3. `early_stopping_min_delta: 0.0`

## Known Notes

1. The repo is intentionally early-stage.
2. Structure is prepared for multiple embedding families, not only NPLM.
3. Additional datasets and model families will be added incrementally.

## Roadmap

Planned additions:

1. Word2Vec (CBOW/Skip-gram) pipeline
2. GloVe-style co-occurrence factorization
3. FastText-style subword embeddings
4. LSTM/Transformer LM embeddings
5. Sentence embedding objectives (contrastive/triplet/dual-encoder)
6. Evaluation suite (word similarity, retrieval, transfer)
7. Unified experiment tracking and reproducibility tooling
