#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Ensure repo-root imports work when invoked as "python scripts/use_nplm.py".
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.nplm import NeuralProbabilisticLanguageModel


def _find_latest_checkpoint(artifacts_dir: Path) -> Path:
    candidates = sorted(artifacts_dir.rglob("best_model_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint files matching 'best_model_*.pt' found under: {artifacts_dir}"
        )
    return candidates[-1]


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested cuda, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model_payload(checkpoint_path: Path, device: torch.device) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload type: {type(payload).__name__}")
    required = ("model_config", "model_state_dict", "vocab")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")
    return payload


def _build_model(payload: dict[str, object], device: torch.device) -> NeuralProbabilisticLanguageModel:
    model_config = payload["model_config"]
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint['model_config'] must be a mapping")
    model = NeuralProbabilisticLanguageModel(
        vocab_size=int(model_config["vocab_size"]),
        context_size=int(model_config["context_size"]),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        use_direct_connection=bool(model_config["use_direct_connection"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _prepare_vocab(payload: dict[str, object]) -> tuple[dict[str, int], list[str], int]:
    vocab = payload["vocab"]
    if not isinstance(vocab, dict):
        raise ValueError("checkpoint['vocab'] must be a mapping")
    token_to_id = vocab["token_to_id"]
    id_to_token = vocab["id_to_token"]
    unk_token = vocab.get("unk_token", "<unk>")
    if not isinstance(token_to_id, dict) or not isinstance(id_to_token, list):
        raise ValueError("Invalid vocab format in checkpoint")
    if unk_token not in token_to_id:
        raise ValueError(f"unk token '{unk_token}' not present in token_to_id")
    return token_to_id, id_to_token, int(token_to_id[unk_token])


def _context_ids(
    prompt_tokens: list[str],
    token_to_id: dict[str, int],
    unk_id: int,
    context_size: int,
) -> torch.Tensor:
    encoded = [int(token_to_id.get(token, unk_id)) for token in prompt_tokens]
    if len(encoded) < context_size:
        encoded = [unk_id] * (context_size - len(encoded)) + encoded
    else:
        encoded = encoded[-context_size:]
    return torch.tensor([encoded], dtype=torch.long)


def _top_k_next_tokens(
    model: NeuralProbabilisticLanguageModel,
    context: torch.Tensor,
    id_to_token: list[str],
    top_k: int,
    device: torch.device,
    disallow_token_id: int | None = None,
) -> list[tuple[str, float]]:
    with torch.inference_mode():
        context = context.to(device)
        probs = torch.softmax(model(context), dim=-1)[0]
        if disallow_token_id is not None and 0 <= disallow_token_id < probs.numel():
            probs = probs.clone()
            probs[disallow_token_id] = 0.0
            total = float(probs.sum().item())
            if total > 0.0:
                probs = probs / total
        k = min(top_k, probs.numel())
        values, indices = torch.topk(probs, k=k)
        return [(id_to_token[int(idx)], float(val)) for val, idx in zip(values, indices)]


def _generate(
    model: NeuralProbabilisticLanguageModel,
    prompt_tokens: list[str],
    token_to_id: dict[str, int],
    id_to_token: list[str],
    unk_id: int,
    context_size: int,
    steps: int,
    temperature: float,
    sample: bool,
    device: torch.device,
    disallow_token_id: int | None = None,
) -> list[str]:
    tokens = list(prompt_tokens)
    with torch.inference_mode():
        for _ in range(steps):
            context = _context_ids(tokens, token_to_id, unk_id, context_size).to(device)
            logits = model(context)[0]
            if disallow_token_id is not None and 0 <= disallow_token_id < logits.numel():
                logits = logits.clone()
                logits[disallow_token_id] = float("-inf")
            if sample:
                scaled = logits / max(temperature, 1e-6)
                probs = torch.softmax(scaled, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_id = int(torch.argmax(logits, dim=-1).item())
            tokens.append(id_to_token[next_id])
    return tokens


def _sample_token_ids(
    vocab_size: int,
    k: int,
    seed: int,
    exclude_token_id: int | None = None,
) -> torch.Tensor:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    candidate_ids = torch.arange(vocab_size, dtype=torch.long)
    if exclude_token_id is not None and 0 <= exclude_token_id < vocab_size:
        candidate_ids = candidate_ids[candidate_ids != exclude_token_id]
    if candidate_ids.numel() == 0:
        raise ValueError("No candidate tokens available after exclusions.")

    k_effective = min(k, int(candidate_ids.numel()))
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(candidate_ids.numel(), generator=generator)
    chosen = candidate_ids[perm[:k_effective]]
    return chosen


def _plot_embeddings_pca(
    embedding_weight: torch.Tensor,
    chosen_ids: torch.Tensor,
    id_to_token: list[str],
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = embedding_weight[chosen_ids].detach().cpu().float()
    if selected.size(0) == 1:
        coords = torch.zeros((1, 2), dtype=torch.float32)
    else:
        centered = selected - selected.mean(dim=0, keepdim=True)
        if min(centered.shape[0], centered.shape[1]) >= 2:
            _, _, v = torch.pca_lowrank(centered, q=2, center=False)
            coords = centered @ v[:, :2]
        else:
            first_dim = centered[:, :1] if centered.shape[1] > 0 else torch.zeros((centered.shape[0], 1))
            coords = torch.cat([first_dim, torch.zeros_like(first_dim)], dim=1)

    x = coords[:, 0].numpy()
    y = coords[:, 1].numpy()
    labels = [id_to_token[int(idx)] for idx in chosen_ids.tolist()]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x, y, s=24, alpha=0.9)
    for token, x_val, y_val in zip(labels, x, y):
        ax.annotate(token, (x_val, y_val), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_title("Token Embeddings: First Two Principal Components")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_embeddings_distance_matrix(
    embedding_weight: torch.Tensor,
    chosen_ids: torch.Tensor,
    id_to_token: list[str],
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = embedding_weight[chosen_ids].detach().cpu().float()
    dist = torch.cdist(selected, selected, p=2).numpy()
    labels = [id_to_token[int(idx)] for idx in chosen_ids.tolist()]

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(dist, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="L2 distance")
    ax.set_title("Token Embedding Distance Matrix")
    ax.set_xlabel("Token")
    ax.set_ylabel("Token")

    if len(labels) <= 40:
        positions = list(range(len(labels)))
        ax.set_xticks(positions)
        ax.set_yticks(positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use a trained NPLM checkpoint for next-token prediction and generation."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint .pt file. If omitted, uses latest best_model_*.pt under --artifacts-dir.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Root directory for automatic checkpoint discovery (default: artifacts).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Space-separated prompt text (tokenization uses str.split()).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Show top-k next-token predictions.")
    parser.add_argument(
        "--generate",
        type=int,
        default=0,
        help="Number of tokens to generate after the prompt.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use stochastic sampling for generation (default: greedy).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature when --sample is set.",
    )
    parser.add_argument(
        "--no-unk",
        action="store_true",
        help="Disallow '<unk>' token in top-k output and generation.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--plot-k",
        type=int,
        default=0,
        help="If > 0, sample k random tokens and save PCA + distance-matrix embedding plots.",
    )
    parser.add_argument(
        "--plot-seed",
        type=int,
        default=42,
        help="Random seed used for token sampling in embedding plots.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("artifacts/nplm/plots"),
        help="Directory to save embedding plot images.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.generate < 0:
        raise ValueError("--generate must be >= 0")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.plot_k < 0:
        raise ValueError("--plot-k must be >= 0")

    checkpoint_path = args.checkpoint or _find_latest_checkpoint(args.artifacts_dir)
    device = _resolve_device(args.device)
    payload = _load_model_payload(checkpoint_path, device=device)
    model = _build_model(payload, device=device)
    token_to_id, id_to_token, unk_id = _prepare_vocab(payload)
    context_size = int(model.context_size)
    disallow_token_id = unk_id if args.no_unk else None

    prompt_tokens = args.prompt.split()
    if not prompt_tokens:
        raise ValueError("Prompt cannot be empty after splitting.")

    context = _context_ids(prompt_tokens, token_to_id, unk_id, context_size)
    topk = _top_k_next_tokens(
        model=model,
        context=context,
        id_to_token=id_to_token,
        top_k=args.top_k,
        device=device,
        disallow_token_id=disallow_token_id,
    )

    print(f"[info] checkpoint={checkpoint_path}")
    print(f"[info] device={device}")
    print(f"[info] context_size={context_size}")
    print(f"[info] prompt={' '.join(prompt_tokens)}")
    print("[next-token top-k]")
    for rank, (token, prob) in enumerate(topk, start=1):
        print(f"{rank:02d}. token={token!r} prob={prob:.6f}")

    if args.generate > 0:
        generated = _generate(
            model=model,
            prompt_tokens=prompt_tokens,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            unk_id=unk_id,
            context_size=context_size,
            steps=args.generate,
            temperature=args.temperature,
            sample=args.sample,
            device=device,
            disallow_token_id=disallow_token_id,
        )
        print("[generated]")
        print(" ".join(generated))

    if args.plot_k > 0:
        embedding_weight = model.embedding.weight.detach().cpu()
        chosen_ids = _sample_token_ids(
            vocab_size=embedding_weight.size(0),
            k=args.plot_k,
            seed=args.plot_seed,
            exclude_token_id=disallow_token_id,
        )
        checkpoint_stem = checkpoint_path.stem
        pca_path = args.plot_dir / f"{checkpoint_stem}_k{chosen_ids.numel()}_pca.png"
        dist_path = args.plot_dir / f"{checkpoint_stem}_k{chosen_ids.numel()}_dist.png"
        _plot_embeddings_pca(
            embedding_weight=embedding_weight,
            chosen_ids=chosen_ids,
            id_to_token=id_to_token,
            output_path=pca_path,
        )
        _plot_embeddings_distance_matrix(
            embedding_weight=embedding_weight,
            chosen_ids=chosen_ids,
            id_to_token=id_to_token,
            output_path=dist_path,
        )
        print("[plots]")
        print(f"pca={pca_path}")
        print(f"distance_matrix={dist_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
