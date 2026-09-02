"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .bootstrap import capture_trajectory
from .config import load_config
from .llama_backend import LlamaBackend
from .search import SearchRunner
from .trajectory import Trajectory


def _bootstrap(config_path: str) -> int:
    config = load_config(config_path)
    config.search.output_dir.mkdir(parents=True, exist_ok=True)
    backend = LlamaBackend(config.model)
    try:
        trajectory = capture_trajectory(config, backend)
        destination = config.search.output_dir / "baseline.json"
        trajectory.save(destination)
        hop2 = backend.score_next(
            trajectory.hop2_context_ids, trajectory.target_eog_id
        )
        summary = {
            "baseline": str(destination),
            "prompt_tokens": len(trajectory.prompt_token_ids),
            "hop1_tokens": len(trajectory.hop1_output_ids),
            "hop1_text": trajectory.hop1_visible_text,
            "target_eog_id": trajectory.target_eog_id,
            "hop2": hop2.as_dict(),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    finally:
        backend.close()


def _search(config_path: str, baseline_path: str | None) -> int:
    config = load_config(config_path)
    source = (
        Path(baseline_path).expanduser().resolve()
        if baseline_path
        else config.search.output_dir / "baseline.json"
    )
    trajectory = Trajectory.load(source)
    backend = LlamaBackend(config.model)
    try:
        result = SearchRunner(config, trajectory, backend).run()
        return 0 if result["solved"] else 2
    finally:
        backend.close()


def _inspect(path: str) -> int:
    trajectory = Trajectory.load(path)
    summary = {
        "gguf": trajectory.gguf_name,
        "gguf_size_bytes": trajectory.gguf_size_bytes,
        "gguf_vocab_size": trajectory.gguf_vocab_size,
        "llama_cpp_version": trajectory.llama_cpp_version,
        "prompt_tokens": len(trajectory.prompt_token_ids),
        "hop1_tokens": len(trajectory.hop1_output_ids),
        "hop1_text": trajectory.hop1_visible_text,
        "hop2_context_tokens": len(trajectory.hop2_context_ids),
        "target_eog": trajectory.target_eog_text,
        "target_eog_id": trajectory.target_eog_id,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hybrid-gcg",
        description="BF16 HotFlip proposals with exact GGUF trajectory gates",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="Decode and save the hop-1 trajectory to preserve"
    )
    bootstrap.add_argument("--config", required=True)

    search = subparsers.add_parser("search", help="Optimize immediate hop-2 EOG")
    search.add_argument("--config", required=True)
    search.add_argument("--baseline")

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect a baseline without loading either model"
    )
    inspect_parser.add_argument("--baseline", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "bootstrap":
        return _bootstrap(args.config)
    if args.command == "search":
        return _search(args.config, args.baseline)
    if args.command == "inspect":
        return _inspect(args.baseline)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
