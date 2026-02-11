#!/usr/bin/env python3
"""Download Penn Treebank language modeling text splits.

This script downloads:
- ptb.train.txt
- ptb.valid.txt
- ptb.test.txt

Default output directory: data/ptb
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path


PTB_MIRRORS: dict[str, list[str]] = {
    "ptb.train.txt": [
        "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
        "https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/penn/train.txt",
    ],
    "ptb.valid.txt": [
        "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
        "https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/penn/valid.txt",
    ],
    "ptb.test.txt": [
        "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
        "https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/penn/test.txt",
    ],
}


def _download_bytes(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "cool-embeddings/ptb-downloader"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_file(filename: str, urls: list[str], output_dir: Path, timeout: float) -> None:
    errors: list[str] = []
    for url in urls:
        try:
            data = _download_bytes(url, timeout=timeout)
            destination = output_dir / filename
            destination.write_bytes(data)
            print(f"[ok] {filename} ({len(data)} bytes) <- {url}")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            errors.append(f"{url} -> {exc}")

    error_text = "\n".join(errors)
    raise RuntimeError(f"Failed to download {filename} from all mirrors:\n{error_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PTB train/valid/test text files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ptb"),
        help="Directory to write PTB files (default: data/ptb).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Output directory: {output_dir.resolve()}")
    failures = 0

    for filename, urls in PTB_MIRRORS.items():
        destination = output_dir / filename
        if destination.exists() and not args.force:
            print(f"[skip] {filename} already exists (use --force to redownload)")
            continue

        try:
            download_file(filename=filename, urls=urls, output_dir=output_dir, timeout=args.timeout)
        except RuntimeError as exc:
            failures += 1
            print(f"[err] {exc}", file=sys.stderr)

    if failures > 0:
        print(f"[done] Completed with {failures} failure(s).", file=sys.stderr)
        return 1

    print("[done] PTB dataset is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
