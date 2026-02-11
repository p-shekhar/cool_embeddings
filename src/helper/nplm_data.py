from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


SPLITS: tuple[str, ...] = ("train", "valid", "test")
PTB_FILENAMES: dict[str, str] = {
    "train": "ptb.train.txt",
    "valid": "ptb.valid.txt",
    "test": "ptb.test.txt",
}
UNK_TOKEN = "<unk>"


class DatasetProvider(Protocol):
    def load_splits(self) -> dict[str, list[str]]:
        """Return tokenized text for train/valid/test splits."""


def _read_tokens(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    text = path.read_text(encoding="utf-8")
    return text.split()


@dataclass
class PTBDatasetProvider:
    data_dir: Path

    def load_splits(self) -> dict[str, list[str]]:
        splits: dict[str, list[str]] = {}
        for split, filename in PTB_FILENAMES.items():
            path = self.data_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing PTB split file: {path}")
            splits[split] = _read_tokens(path)
        return splits


@dataclass
class TextFilesDatasetProvider:
    train_file: Path
    valid_file: Path
    test_file: Path

    def load_splits(self) -> dict[str, list[str]]:
        return {
            "train": _read_tokens(self.train_file),
            "valid": _read_tokens(self.valid_file),
            "test": _read_tokens(self.test_file),
        }


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    unk_token: str = UNK_TOKEN

    @classmethod
    def build(cls, train_tokens: list[str], unk_token: str = UNK_TOKEN) -> Vocabulary:
        unique_tokens = sorted(set(train_tokens))
        if unk_token in unique_tokens:
            id_to_token = unique_tokens
        else:
            id_to_token = [unk_token] + unique_tokens
        token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token, unk_token=unk_token)

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def __len__(self) -> int:
        return len(self.id_to_token)


class NgramLanguageModelDataset(Dataset[tuple[Tensor, Tensor]]):
    """Build (context, target) pairs from a token id sequence."""

    def __init__(self, token_ids: list[int], context_size: int) -> None:
        if context_size <= 0:
            raise ValueError("context_size must be > 0")
        if len(token_ids) <= context_size:
            raise ValueError(
                f"Not enough tokens ({len(token_ids)}) for context_size={context_size}"
            )
        self.context_size = context_size
        self.token_ids = token_ids

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_size

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        context = self.token_ids[idx : idx + self.context_size]
        target = self.token_ids[idx + self.context_size]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def create_provider(args: argparse.Namespace) -> DatasetProvider:
    if args.dataset == "ptb":
        return PTBDatasetProvider(data_dir=args.data_dir)
    if args.dataset == "text-files":
        if not args.train_file or not args.valid_file or not args.test_file:
            raise ValueError(
                "--train-file, --valid-file, and --test-file are required for dataset=text-files"
            )
        return TextFilesDatasetProvider(
            train_file=args.train_file,
            valid_file=args.valid_file,
            test_file=args.test_file,
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_dataloaders(
    splits: dict[str, list[int]],
    context_size: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader[tuple[Tensor, Tensor]]]:
    datasets = {
        split: NgramLanguageModelDataset(token_ids=token_ids, context_size=context_size)
        for split, token_ids in splits.items()
    }
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
