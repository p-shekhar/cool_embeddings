from __future__ import annotations

import argparse
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path("configs/config_nplm.yaml")
DEFAULT_LOG_DIR = Path("logs")


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config file root must be a mapping/object.")
    return data


def _section(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping/object.")
    return value


def config_defaults(config: dict[str, object]) -> dict[str, object]:
    data_cfg = _section(config, "data")
    model_cfg = _section(config, "model")
    optim_cfg = _section(config, "optimization")
    runtime_cfg = _section(config, "runtime")
    output_cfg = _section(config, "output")
    logging_cfg = _section(config, "logging")
    train_file_value = data_cfg.get("train_file", config.get("train_file"))
    valid_file_value = data_cfg.get("valid_file", config.get("valid_file"))
    test_file_value = data_cfg.get("test_file", config.get("test_file"))

    return {
        "dataset": data_cfg.get("dataset", config.get("dataset", "ptb")),
        "data_dir": Path(str(data_cfg.get("data_dir", config.get("data_dir", "data/ptb")))),
        "train_file": Path(str(train_file_value)) if train_file_value else None,
        "valid_file": Path(str(valid_file_value)) if valid_file_value else None,
        "test_file": Path(str(test_file_value)) if test_file_value else None,
        "context_size": int(model_cfg.get("context_size", config.get("context_size", 4))),
        "embedding_dim": int(model_cfg.get("embedding_dim", config.get("embedding_dim", 128))),
        "hidden_dim": int(model_cfg.get("hidden_dim", config.get("hidden_dim", 256))),
        "direct_connection": bool(
            model_cfg.get("direct_connection", config.get("direct_connection", True))
        ),
        "epochs": int(optim_cfg.get("epochs", config.get("epochs", 10))),
        "batch_size": int(optim_cfg.get("batch_size", config.get("batch_size", 256))),
        "lr": float(optim_cfg.get("lr", config.get("lr", 1e-3))),
        "weight_decay": float(optim_cfg.get("weight_decay", config.get("weight_decay", 0.0))),
        "num_workers": int(optim_cfg.get("num_workers", config.get("num_workers", 0))),
        "seed": int(optim_cfg.get("seed", config.get("seed", 42))),
        "device": str(runtime_cfg.get("device", config.get("device", "auto"))),
        "save_path": Path(
            str(output_cfg.get("save_path", config.get("save_path", "artifacts/nplm/best_model.pt")))
        ),
        "metrics_path": Path(
            str(
                output_cfg.get(
                    "metrics_path",
                    config.get("metrics_path", "artifacts/nplm/metrics.json"),
                )
            )
        ),
        "log_dir": Path(str(logging_cfg.get("log_dir", config.get("log_dir", str(DEFAULT_LOG_DIR))))),
        "log_file": logging_cfg.get("log_file", config.get("log_file")),
    }


def add_cli_args(parser: argparse.ArgumentParser, defaults: dict[str, object] | None = None) -> None:
    defaults = defaults or {}
    parser.add_argument("--dataset", choices=("ptb", "text-files"), default=defaults.get("dataset", "ptb"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=defaults.get("data_dir", Path("data/ptb")),
        help="Used when --dataset=ptb.",
    )
    parser.add_argument("--train-file", type=Path, default=defaults.get("train_file"))
    parser.add_argument("--valid-file", type=Path, default=defaults.get("valid_file"))
    parser.add_argument("--test-file", type=Path, default=defaults.get("test_file"))

    parser.add_argument("--context-size", type=int, default=defaults.get("context_size", 4))
    parser.add_argument("--embedding-dim", type=int, default=defaults.get("embedding_dim", 128))
    parser.add_argument("--hidden-dim", type=int, default=defaults.get("hidden_dim", 256))
    parser.add_argument(
        "--direct-connection",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("direct_connection", True),
        help="Enable/disable Bengio direct context-to-output connection.",
    )

    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 10))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 256))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 1e-3))
    parser.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 0.0))
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 0))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--device", default=defaults.get("device", "auto"), choices=("auto", "cpu", "cuda"))
    parser.add_argument("--save-path", type=Path, default=defaults.get("save_path", Path("artifacts/nplm/best_model.pt")))
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=defaults.get("metrics_path", Path("artifacts/nplm/metrics.json")),
    )
    parser.add_argument("--log-dir", type=Path, default=defaults.get("log_dir", DEFAULT_LOG_DIR))
    parser.add_argument("--log-file", default=defaults.get("log_file"))


def _collect_cli_provided_dests(parser: argparse.ArgumentParser, argv: list[str] | None) -> set[str]:
    if not argv:
        return set()
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            if option.startswith("--"):
                option_to_dest[option] = action.dest

    provided: set[str] = set()
    for token in argv:
        if token.startswith("--"):
            key = token.split("=", 1)[0]
            dest = option_to_dest.get(key)
            if dest:
                provided.add(dest)
    return provided


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)

    raw_config = load_config(config_args.config)
    defaults = config_defaults(raw_config)

    parser = argparse.ArgumentParser(description="Train Bengio-style NPLM.")
    parser.add_argument("--config", type=Path, default=config_args.config)
    add_cli_args(parser, defaults=defaults)
    parsed = parser.parse_args(argv)

    cli_provided = _collect_cli_provided_dests(parser, argv)
    cli_overrides: dict[str, object] = {}
    for key in sorted(cli_provided):
        cli_overrides[key] = _json_ready(getattr(parsed, key))

    parsed._config_defaults = {k: _json_ready(v) for k, v in defaults.items()}
    parsed._raw_config = _json_ready(raw_config)
    parsed._cli_overrides = cli_overrides
    parsed._effective_config = {
        key: _json_ready(value) for key, value in vars(parsed).items() if not key.startswith("_")
    }
    return parsed
