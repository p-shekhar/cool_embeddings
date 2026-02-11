from __future__ import annotations

import datetime
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.models.nplm import NeuralProbabilisticLanguageModel
from src.helper.nplm_cli import add_cli_args, parse_args
from src.helper.nplm_data import SPLITS, Vocabulary, build_dataloaders, create_provider
from src.helper.nplm_logging import run_with_tee_logging


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Metrics:
    loss: float
    perplexity: float


class NPLMTrainer:
    def __init__(
        self,
        model: NeuralProbabilisticLanguageModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader: DataLoader[tuple[Tensor, Tensor]]) -> Metrics:
        self.model.train()
        total_loss = 0.0
        total_examples = 0
        for context, target in loader:
            context = context.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(self.model(context), target)
            loss.backward()
            self.optimizer.step()
            batch_size = target.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
        avg_loss = total_loss / max(total_examples, 1)
        return Metrics(loss=avg_loss, perplexity=math.exp(avg_loss))

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader[tuple[Tensor, Tensor]]) -> Metrics:
        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        for context, target in loader:
            context = context.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            loss = self.criterion(self.model(context), target)
            batch_size = target.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
        avg_loss = total_loss / max(total_examples, 1)
        return Metrics(loss=avg_loss, perplexity=math.exp(avg_loss))


def save_checkpoint(
    path: Path,
    model: NeuralProbabilisticLanguageModel,
    vocabulary: Vocabulary,
    args: object,
    best_valid: Metrics,
    test_metrics: Metrics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": model.vocab_size,
            "context_size": model.context_size,
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
            "use_direct_connection": model.use_direct_connection,
        },
        "vocab": {
            "token_to_id": vocabulary.token_to_id,
            "id_to_token": vocabulary.id_to_token,
            "unk_token": vocabulary.unk_token,
        },
        "training_config": {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "metrics": {
            "best_valid_loss": best_valid.loss,
            "best_valid_ppl": best_valid.perplexity,
            "test_loss": test_metrics.loss,
            "test_ppl": test_metrics.perplexity,
        },
    }
    torch.save(payload, path)


def _timestamped_path(path: Path, run_id: str) -> Path:
    """Append run timestamp to filename to avoid overwrite."""
    if path.suffix:
        name = f"{path.stem}_{run_id}{path.suffix}"
    else:
        name = f"{path.name}_{run_id}"
    return path.with_name(name)


def _resolve_artifact_paths(args: object) -> None:
    run_id = getattr(args, "_run_id", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    args.save_path = _timestamped_path(args.save_path, run_id)
    args.metrics_path = _timestamped_path(args.metrics_path, run_id)


def _cuda_is_usable() -> tuple[bool, str | None]:
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    try:
        probe = torch.tensor([1.0], device="cuda")
        _ = (probe + 1).item()
        torch.cuda.synchronize()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        usable, reason = _cuda_is_usable()
        if not usable:
            raise RuntimeError(
                "Requested cuda but CUDA is not usable in this environment. "
                f"Details: {reason}"
            )
        return torch.device("cuda")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    usable, reason = _cuda_is_usable()
    if usable:
        return torch.device("cuda")
    print(
        "[warn] CUDA is detected but not usable for this PyTorch build. "
        f"Falling back to CPU. Details: {reason}"
    )
    return torch.device("cpu")


def run(args: object) -> int:
    _resolve_artifact_paths(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"[info] device={device}")

    provider = create_provider(args)
    raw_splits = provider.load_splits()
    for split in SPLITS:
        print(f"[info] {split} tokens={len(raw_splits[split])}")

    vocabulary = Vocabulary.build(raw_splits["train"])
    encoded_splits = {split: vocabulary.encode(tokens) for split, tokens in raw_splits.items()}
    print(f"[info] vocab_size={len(vocabulary)}")

    dataloaders = build_dataloaders(
        splits=encoded_splits,
        context_size=args.context_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    for split in SPLITS:
        print(f"[info] {split} examples={len(dataloaders[split].dataset)}")

    model = NeuralProbabilisticLanguageModel(
        vocab_size=len(vocabulary),
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        use_direct_connection=args.direct_connection,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = NPLMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )

    best_valid = Metrics(loss=float("inf"), perplexity=float("inf"))
    best_state_dict: dict[str, Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(dataloaders["train"])
        valid_metrics = trainer.evaluate(dataloaders["valid"])
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics.loss:.4f} train_ppl={train_metrics.perplexity:.2f} "
            f"valid_loss={valid_metrics.loss:.4f} valid_ppl={valid_metrics.perplexity:.2f}"
        )
        if valid_metrics.loss < best_valid.loss:
            best_valid = valid_metrics
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    test_metrics = trainer.evaluate(dataloaders["test"])
    print(f"[final] test_loss={test_metrics.loss:.4f} test_ppl={test_metrics.perplexity:.2f}")

    save_checkpoint(
        path=args.save_path,
        model=model,
        vocabulary=vocabulary,
        args=args,
        best_valid=best_valid,
        test_metrics=test_metrics,
    )
    print(f"[saved] checkpoint={args.save_path}")

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "best_valid_loss": best_valid.loss,
        "best_valid_ppl": best_valid.perplexity,
        "test_loss": test_metrics.loss,
        "test_ppl": test_metrics.perplexity,
        "vocab_size": len(vocabulary),
        "context_size": args.context_size,
        "dataset": args.dataset,
    }
    args.metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"[saved] metrics={args.metrics_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args._run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cli_argv = list(argv) if argv is not None else sys.argv[1:]
    return run_with_tee_logging(args=args, argv=cli_argv, run_fn=lambda: run(args))


if __name__ == "__main__":
    raise SystemExit(main())
