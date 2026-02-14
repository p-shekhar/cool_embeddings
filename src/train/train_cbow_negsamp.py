from __future__ import annotations

import datetime
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from src.helper.cbow_negsamp_cli import parse_args
from src.helper.nplm_data import SPLITS, Vocabulary, build_dataloaders, create_provider
from src.helper.nplm_logging import run_with_tee_logging
from src.models.cbow import ContinuousBagOfWords


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Metrics:
    loss: float
    perplexity: float


def build_negative_sampling_probs(
    train_ids: list[int],
    vocab_size: int,
    power: float,
) -> Tensor:
    counts = torch.bincount(torch.tensor(train_ids, dtype=torch.long), minlength=vocab_size).float()
    probs = counts.pow(power)
    if torch.sum(probs) <= 0:
        raise ValueError("Invalid negative-sampling distribution: sum is zero.")
    probs = probs / probs.sum()
    return probs


def sample_negatives(
    probs: Tensor,
    positive_targets: Tensor,
    num_negatives: int,
    generator: torch.Generator,
) -> Tensor:
    batch_size = int(positive_targets.size(0))
    negatives = torch.multinomial(
        probs,
        num_samples=batch_size * num_negatives,
        replacement=True,
        generator=generator,
    ).view(batch_size, num_negatives)

    mask = negatives.eq(positive_targets.unsqueeze(1))
    while bool(mask.any()):
        resampled = torch.multinomial(
            probs,
            num_samples=int(mask.sum().item()),
            replacement=True,
            generator=generator,
        )
        negatives[mask] = resampled
        mask = negatives.eq(positive_targets.unsqueeze(1))
    return negatives


class CBOWNegativeSamplingTrainer:
    def __init__(
        self,
        model: ContinuousBagOfWords,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        negative_probs: Tensor,
        num_negatives: int,
        seed: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.negative_probs = negative_probs.cpu()
        self.num_negatives = num_negatives
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def _batch_loss(self, context: Tensor, positive_target: Tensor) -> Tensor:
        context = context.to(self.device, non_blocking=True)
        positive_target = positive_target.to(self.device, non_blocking=True)

        negatives = sample_negatives(
            probs=self.negative_probs,
            positive_targets=positive_target.detach().cpu(),
            num_negatives=self.num_negatives,
            generator=self.generator,
        ).to(self.device, non_blocking=True)

        context_embed = self.model.embedding(context).mean(dim=1)  # [B, E]
        out_weight = self.model.output.weight  # [V, E]
        out_bias = self.model.output.bias  # [V]

        pos_weight = out_weight[positive_target]  # [B, E]
        pos_bias = out_bias[positive_target]  # [B]
        pos_logits = (context_embed * pos_weight).sum(dim=1) + pos_bias  # [B]

        neg_weight = out_weight[negatives]  # [B, K, E]
        neg_bias = out_bias[negatives]  # [B, K]
        neg_logits = torch.bmm(neg_weight, context_embed.unsqueeze(2)).squeeze(2) + neg_bias  # [B, K]

        pos_loss = F.logsigmoid(pos_logits)
        neg_loss = F.logsigmoid(-neg_logits).sum(dim=1)
        loss = -(pos_loss + neg_loss).mean()
        return loss

    def train_epoch(self, loader: DataLoader[tuple[Tensor, Tensor]]) -> Metrics:
        self.model.train()
        total_loss = 0.0
        total_examples = 0
        for context, target in loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._batch_loss(context, target)
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
            loss = self._batch_loss(context, target)
            batch_size = target.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
        avg_loss = total_loss / max(total_examples, 1)
        return Metrics(loss=avg_loss, perplexity=math.exp(avg_loss))


def save_checkpoint(
    path: Path,
    model: ContinuousBagOfWords,
    vocabulary: Vocabulary,
    args: object,
    best_valid: Metrics,
    test_metrics: Metrics,
    epoch_history: list[dict[str, float | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "model_type": "cbow_negsamp",
            "vocab_size": model.vocab_size,
            "context_size": model.context_size,
            "embedding_dim": model.embedding_dim,
            "num_negatives": args.num_negatives,
            "negative_sampling_power": args.negative_sampling_power,
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
            "num_negatives": args.num_negatives,
            "negative_sampling_power": args.negative_sampling_power,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "seed": args.seed,
        },
        "metrics": {
            "best_valid_loss": best_valid.loss,
            "best_valid_ppl": best_valid.perplexity,
            "test_loss": test_metrics.loss,
            "test_ppl": test_metrics.perplexity,
            "epoch_history": epoch_history,
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

    if args.num_negatives <= 0:
        raise ValueError("num_negatives must be > 0")

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

    negative_probs = build_negative_sampling_probs(
        train_ids=encoded_splits["train"],
        vocab_size=len(vocabulary),
        power=args.negative_sampling_power,
    )

    model = ContinuousBagOfWords(
        vocab_size=len(vocabulary),
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = CBOWNegativeSamplingTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        negative_probs=negative_probs,
        num_negatives=args.num_negatives,
        seed=args.seed,
    )

    best_valid = Metrics(loss=float("inf"), perplexity=float("inf"))
    best_state_dict: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    epoch_history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(dataloaders["train"])
        valid_metrics = trainer.evaluate(dataloaders["valid"])
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_ppl": train_metrics.perplexity,
                "valid_loss": valid_metrics.loss,
                "valid_ppl": valid_metrics.perplexity,
            }
        )
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics.loss:.4f} train_ppl={train_metrics.perplexity:.2f} "
            f"valid_loss={valid_metrics.loss:.4f} valid_ppl={valid_metrics.perplexity:.2f}"
        )
        if valid_metrics.loss < (best_valid.loss - args.early_stopping_min_delta):
            best_valid = valid_metrics
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and (
                epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    "[early-stop] "
                    f"no validation improvement for {epochs_without_improvement} epoch(s); "
                    f"patience={args.early_stopping_patience}"
                )
                break

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
        epoch_history=epoch_history,
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
        "embedding_dim": args.embedding_dim,
        "num_negatives": args.num_negatives,
        "negative_sampling_power": args.negative_sampling_power,
        "dataset": args.dataset,
        "epoch_history": epoch_history,
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
