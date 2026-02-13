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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.command == "train-nplm":
        from src.train import train_nplm

        if args.train_help:
            return train_nplm.main(["--help"])
        return train_nplm.main(remaining)
    if args.command == "use-nplm":
        from scripts import use_nplm

        if args.use_help:
            return use_nplm.main(["--help"])
        return use_nplm.main(remaining)
    parser.error(f"Unknown command: {args.command}")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
