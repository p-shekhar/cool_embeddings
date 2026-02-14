from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="cool-embeddings CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train-nplm",
        add_help=False,
        help="Train Bengio-style Neural Probabilistic Language Model.",
    )
    train_parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="train_help",
        help="Show train-nplm help message.",
    )
    train_cbow_parser = subparsers.add_parser(
        "train-cbow",
        add_help=False,
        help="Train Continuous Bag-of-Words (CBOW) model.",
    )
    train_cbow_parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="train_cbow_help",
        help="Show train-cbow help message.",
    )
    train_cbow_negsamp_parser = subparsers.add_parser(
        "train-cbow-negsamp",
        add_help=False,
        help="Train CBOW model with negative sampling.",
    )
    train_cbow_negsamp_parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="train_cbow_negsamp_help",
        help="Show train-cbow-negsamp help message.",
    )
    train_skipgram_parser = subparsers.add_parser(
        "train-skipgram",
        add_help=False,
        help="Train Skip-gram model.",
    )
    train_skipgram_parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="train_skipgram_help",
        help="Show train-skipgram help message.",
    )
    train_skipgram_negsamp_parser = subparsers.add_parser(
        "train-skipgram-negsamp",
        add_help=False,
        help="Train Skip-gram model with negative sampling.",
    )
    train_skipgram_negsamp_parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="train_skipgram_negsamp_help",
        help="Show train-skipgram-negsamp help message.",
    )

    use_parser = subparsers.add_parser(
        "use-nplm",
        add_help=False,
        help="Use a trained NPLM checkpoint for prediction/generation.",
    )
    use_parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="use_help",
        help="Show use-nplm help message.",
    )
    use_cbow_parser = subparsers.add_parser(
        "use-cbow",
        add_help=False,
        help="Use a trained CBOW checkpoint for embedding queries.",
    )
    use_cbow_parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="use_cbow_help",
        help="Show use-cbow help message.",
    )
    use_skipgram_parser = subparsers.add_parser(
        "use-skipgram",
        add_help=False,
        help="Use trained Skip-gram checkpoints for embedding queries.",
    )
    use_skipgram_parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="use_skipgram_help",
        help="Show use-skipgram help message.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.command == "train-nplm":
        from src.train import train_nplm

        if args.train_help:
            return train_nplm.main(["--help"])
        return train_nplm.main(remaining)
    if args.command == "train-cbow":
        from src.train import train_cbow

        if args.train_cbow_help:
            return train_cbow.main(["--help"])
        return train_cbow.main(remaining)
    if args.command == "train-cbow-negsamp":
        from src.train import train_cbow_negsamp

        if args.train_cbow_negsamp_help:
            return train_cbow_negsamp.main(["--help"])
        return train_cbow_negsamp.main(remaining)
    if args.command == "train-skipgram":
        from src.train import train_skipgram

        if args.train_skipgram_help:
            return train_skipgram.main(["--help"])
        return train_skipgram.main(remaining)
    if args.command == "train-skipgram-negsamp":
        from src.train import train_skipgram_negsamp

        if args.train_skipgram_negsamp_help:
            return train_skipgram_negsamp.main(["--help"])
        return train_skipgram_negsamp.main(remaining)
    if args.command == "use-nplm":
        from scripts import use_nplm

        if args.use_help:
            return use_nplm.main(["--help"])
        return use_nplm.main(remaining)
    if args.command == "use-cbow":
        from scripts import use_cbow

        if args.use_cbow_help:
            return use_cbow.main(["--help"])
        return use_cbow.main(remaining)
    if args.command == "use-skipgram":
        from scripts import use_skipgram

        if args.use_skipgram_help:
            return use_skipgram.main(["--help"])
        return use_skipgram.main(remaining)
    parser.error(f"Unknown command: {args.command}")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
