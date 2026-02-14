# Project Direction

`cool-embeddings` is a learning-first, implementation-driven repository for embedding models.

## Goal

Build one codebase that tracks the evolution of embeddings:

1. Classic feedforward neural language models
2. Word embeddings (Word2Vec/GloVe/FastText-style workflows)
3. Contextual sequence models (RNN/LSTM/Transformer encoders)
4. Contrastive and sentence embeddings
5. Modern large-scale embedding approaches

## Current Status

Implemented now:

1. NPLM in PyTorch (`src/models/nplm.py`)
2. CBOW in PyTorch (`src/models/cbow.py`)
3. CBOW negative-sampling training pipeline
4. Skip-gram in PyTorch (`src/models/skipgram.py`)
5. Skip-gram negative-sampling training pipeline
6. PTB downloader
7. Config-driven training with CLI overrides
8. Per-run logging
9. Timestamped artifacts to avoid overwrite
10. Per-epoch metric history saved for train/valid loss and perplexity

## Implemented CLI Surface

1. `train-nplm`: train feedforward language model
2. `use-nplm`: LM inference + embedding inspection/similarity/analogy/plots/export + training-curve plots
3. `train-cbow`: train CBOW embedding model
4. `train-cbow-negsamp`: train CBOW with negative sampling
5. `use-cbow`: embedding inspection/similarity/analogy/plots/export + training-curve plots
6. `train-skipgram`: train skip-gram with full softmax objective
7. `train-skipgram-negsamp`: train skip-gram with negative sampling
8. `use-skipgram`: embedding inspection/similarity/analogy/plots/export + training-curve plots

## Notes

1. The repo is intentionally early-stage.
2. Structure is prepared for multiple embedding families.
3. Additional datasets and model families will be added incrementally.

## Roadmap

Planned additions:

1. GloVe-style co-occurrence factorization
2. FastText-style subword embeddings
3. LSTM/Transformer LM embeddings
4. Sentence embedding objectives (contrastive/triplet/dual-encoder)
5. Evaluation suite (word similarity, retrieval, transfer)
6. Unified experiment tracking and reproducibility tooling
