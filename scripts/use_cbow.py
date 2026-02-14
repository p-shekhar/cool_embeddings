#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Ensure repo-root imports work when invoked as "python scripts/use_cbow.py".
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.cbow import ContinuousBagOfWords


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


def _build_model(payload: dict[str, object], device: torch.device) -> ContinuousBagOfWords:
    model_config = payload["model_config"]
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint['model_config'] must be a mapping")
    model = ContinuousBagOfWords(
        vocab_size=int(model_config["vocab_size"]),
        context_size=int(model_config["context_size"]),
        embedding_dim=int(model_config["embedding_dim"]),
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


def _token_id(token: str, token_to_id: dict[str, int], unk_id: int) -> int:
    return int(token_to_id.get(token, unk_id))


def _cosine_similarity(embeddings: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    e_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp_min(eps)
    v_norm = vec / vec.norm().clamp_min(eps)
    return e_norm @ v_norm


def _nearest_neighbors(
    token: str,
    embeddings: torch.Tensor,
    token_to_id: dict[str, int],
    id_to_token: list[str],
    unk_id: int,
    top_k: int,
    no_unk: bool,
) -> list[tuple[str, float]]:
    query_id = _token_id(token, token_to_id, unk_id)
    query_vec = embeddings[query_id]
    scores = _cosine_similarity(embeddings, query_vec)

    scores = scores.clone()
    scores[query_id] = -1.0
    if no_unk and 0 <= unk_id < scores.numel():
        scores[unk_id] = -1.0

    k = min(top_k, int(scores.numel()))
    values, indices = torch.topk(scores, k=k)
    return [(id_to_token[int(i)], float(v)) for v, i in zip(values, indices)]


def _pair_similarity(
    token_a: str,
    token_b: str,
    embeddings: torch.Tensor,
    token_to_id: dict[str, int],
    unk_id: int,
) -> float:
    id_a = _token_id(token_a, token_to_id, unk_id)
    id_b = _token_id(token_b, token_to_id, unk_id)
    sim = _cosine_similarity(embeddings, embeddings[id_a])[id_b]
    return float(sim)


def _analogy_neighbors(
    token_a: str,
    token_b: str,
    token_c: str,
    embeddings: torch.Tensor,
    token_to_id: dict[str, int],
    id_to_token: list[str],
    unk_id: int,
    top_k: int,
    no_unk: bool,
) -> list[tuple[str, float]]:
    id_a = _token_id(token_a, token_to_id, unk_id)
    id_b = _token_id(token_b, token_to_id, unk_id)
    id_c = _token_id(token_c, token_to_id, unk_id)

    target = embeddings[id_a] - embeddings[id_b] + embeddings[id_c]
    scores = _cosine_similarity(embeddings, target).clone()

    for idx in (id_a, id_b, id_c):
        if 0 <= idx < scores.numel():
            scores[idx] = -1.0
    if no_unk and 0 <= unk_id < scores.numel():
        scores[unk_id] = -1.0

    k = min(top_k, int(scores.numel()))
    values, indices = torch.topk(scores, k=k)
    return [(id_to_token[int(i)], float(v)) for v, i in zip(values, indices)]


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
    return candidate_ids[perm[:k_effective]]


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
            first_dim = (
                centered[:, :1]
                if centered.shape[1] > 0
                else torch.zeros((centered.shape[0], 1))
            )
            coords = torch.cat([first_dim, torch.zeros_like(first_dim)], dim=1)

    x = coords[:, 0].numpy()
    y = coords[:, 1].numpy()
    labels = [id_to_token[int(idx)] for idx in chosen_ids.tolist()]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x, y, s=24, alpha=0.9)
    for token, x_val, y_val in zip(labels, x, y):
        ax.annotate(token, (x_val, y_val), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_title("CBOW Embeddings: PCA (2D)")
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
    ax.set_title("CBOW Token Embedding Distance Matrix")
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
        description="Use a trained CBOW checkpoint for embedding queries and plots."
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
        default=Path("artifacts/cbow"),
        help="Root directory for automatic checkpoint discovery (default: artifacts/cbow).",
    )
    parser.add_argument("--token", type=str, default=None, help="Show embedding summary for a token.")
    parser.add_argument(
        "--show-n",
        type=int,
        default=8,
        help="How many leading dimensions to print for --token (default: 8).",
    )
    parser.add_argument(
        "--similar",
        type=str,
        default=None,
        help="Show nearest neighbors by cosine similarity for a token.",
    )
    parser.add_argument(
        "--pair-sim",
        nargs=2,
        metavar=("TOKEN_A", "TOKEN_B"),
        default=None,
        help="Cosine similarity between two tokens.",
    )
    parser.add_argument(
        "--analogy",
        nargs=3,
        metavar=("A", "B", "C"),
        default=None,
        help="Word analogy query: A - B + C, return top-k nearest tokens.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for --similar results.")
    parser.add_argument(
        "--no-unk",
        action="store_true",
        help="Exclude '<unk>' from nearest-neighbor search and plots.",
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
        default=Path("artifacts/cbow/plots"),
        help="Directory to save embedding plot images.",
    )
    parser.add_argument(
        "--export-embeddings",
        type=Path,
        default=None,
        help="Optional .pt path to export full embedding table payload.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.show_n <= 0:
        raise ValueError("--show-n must be > 0")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.plot_k < 0:
        raise ValueError("--plot-k must be >= 0")

    has_operation = any(
        [
            args.token is not None,
            args.similar is not None,
            args.pair_sim is not None,
            args.analogy is not None,
            args.plot_k > 0,
            args.export_embeddings is not None,
        ]
    )
    if not has_operation:
        parser.error(
            "Provide at least one operation: "
            "--token/--similar/--pair-sim/--analogy/--plot-k/--export-embeddings"
        )

    checkpoint_path = args.checkpoint or _find_latest_checkpoint(args.artifacts_dir)
    device = _resolve_device(args.device)
    payload = _load_model_payload(checkpoint_path, device=device)
    model = _build_model(payload, device=device)
    token_to_id, id_to_token, unk_id = _prepare_vocab(payload)
    embeddings = model.embedding_weight().detach().cpu()
    excluded_token_id = unk_id if args.no_unk else None

    print(f"[info] checkpoint={checkpoint_path}")
    print(f"[info] device={device}")
    print(f"[info] vocab_size={embeddings.size(0)}")
    print(f"[info] embedding_dim={embeddings.size(1)}")

    if args.token is not None:
        tid = _token_id(args.token, token_to_id, unk_id)
        resolved = id_to_token[tid]
        vec = embeddings[tid]
        preview_len = min(args.show_n, int(vec.numel()))
        preview = ", ".join(f"{float(v):.5f}" for v in vec[:preview_len])
        print("[token]")
        print(f"query={args.token!r} resolved={resolved!r} id={tid}")
        print(f"l2_norm={float(torch.linalg.vector_norm(vec)):.6f}")
        print(f"first_{preview_len}=[{preview}]")

    if args.similar is not None:
        neighbors = _nearest_neighbors(
            token=args.similar,
            embeddings=embeddings,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            unk_id=unk_id,
            top_k=args.top_k,
            no_unk=args.no_unk,
        )
        print("[similar]")
        print(f"query={args.similar!r}")
        for rank, (token, score) in enumerate(neighbors, start=1):
            print(f"{rank:02d}. token={token!r} cosine={score:.6f}")

    if args.pair_sim is not None:
        token_a, token_b = args.pair_sim
        score = _pair_similarity(
            token_a=token_a,
            token_b=token_b,
            embeddings=embeddings,
            token_to_id=token_to_id,
            unk_id=unk_id,
        )
        print("[pair-sim]")
        print(f"token_a={token_a!r} token_b={token_b!r} cosine={score:.6f}")

    if args.analogy is not None:
        token_a, token_b, token_c = args.analogy
        neighbors = _analogy_neighbors(
            token_a=token_a,
            token_b=token_b,
            token_c=token_c,
            embeddings=embeddings,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            unk_id=unk_id,
            top_k=args.top_k,
            no_unk=args.no_unk,
        )
        print("[analogy]")
        print(f"query={token_a!r} - {token_b!r} + {token_c!r}")
        for rank, (token, score) in enumerate(neighbors, start=1):
            print(f"{rank:02d}. token={token!r} cosine={score:.6f}")

    if args.plot_k > 0:
        chosen_ids = _sample_token_ids(
            vocab_size=int(embeddings.size(0)),
            k=args.plot_k,
            seed=args.plot_seed,
            exclude_token_id=excluded_token_id,
        )
        checkpoint_stem = checkpoint_path.stem
        pca_path = args.plot_dir / f"{checkpoint_stem}_k{chosen_ids.numel()}_pca.png"
        dist_path = args.plot_dir / f"{checkpoint_stem}_k{chosen_ids.numel()}_dist.png"
        _plot_embeddings_pca(
            embedding_weight=embeddings,
            chosen_ids=chosen_ids,
            id_to_token=id_to_token,
            output_path=pca_path,
        )
        _plot_embeddings_distance_matrix(
            embedding_weight=embeddings,
            chosen_ids=chosen_ids,
            id_to_token=id_to_token,
            output_path=dist_path,
        )
        print("[plots]")
        print(f"pca={pca_path}")
        print(f"distance_matrix={dist_path}")

    if args.export_embeddings is not None:
        args.export_embeddings.parent.mkdir(parents=True, exist_ok=True)
        export_payload = {
            "checkpoint": str(checkpoint_path),
            "vocab": {
                "token_to_id": token_to_id,
                "id_to_token": id_to_token,
                "unk_id": unk_id,
            },
            "embeddings": embeddings,
        }
        torch.save(export_payload, args.export_embeddings)
        print("[export]")
        print(f"path={args.export_embeddings}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
