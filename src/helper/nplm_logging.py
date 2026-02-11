from __future__ import annotations

import contextlib
import datetime
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Callable


class TeeWriter(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            try:
                if getattr(stream, "closed", False):
                    continue
                stream.write(data)
                stream.flush()
            except ValueError:
                continue
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                if getattr(stream, "closed", False):
                    continue
                stream.flush()
            except ValueError:
                continue


def resolve_log_path(log_dir: Path, log_file: str | None) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_file:
        filename = Path(str(log_file)).name
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{timestamp}.log"
    return log_dir / filename


def log_run_header(args: object, argv: list[str]) -> None:
    now = datetime.datetime.now().isoformat(timespec="seconds")
    print("=" * 80)
    print(f"[run] started_at={now}")
    print(f"[run] argv={' '.join(argv)}")
    print(f"[run] config_file={getattr(args, 'config', '')}")
    print("[run] effective_config=")
    print(json.dumps(getattr(args, "_effective_config", {}), indent=2, sort_keys=True))
    print("[run] cli_overrides=")
    print(json.dumps(getattr(args, "_cli_overrides", {}), indent=2, sort_keys=True))
    print("=" * 80)


def run_with_tee_logging(args: object, argv: list[str], run_fn: Callable[[], int]) -> int:
    log_dir = getattr(args, "log_dir")
    custom_log_file = getattr(args, "log_file")
    if custom_log_file:
        log_path = resolve_log_path(log_dir, custom_log_file)
    else:
        run_id = getattr(args, "_run_id", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        log_path = resolve_log_path(log_dir, f"run_{run_id}.log")
    with log_path.open("a", encoding="utf-8") as log_file:
        tee_out = TeeWriter(sys.stdout, log_file)
        tee_err = TeeWriter(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"[log] writing run logs to {log_path}")
            log_run_header(args, argv)
            try:
                return run_fn()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                return 1
