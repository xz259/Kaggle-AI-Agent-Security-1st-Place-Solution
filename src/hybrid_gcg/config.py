"""Typed TOML configuration for the public runner."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    gguf_path: Path
    hf_model: str
    model_class: str = "AutoModelForCausalLM"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    n_ctx: int = 8192
    n_batch: int = 512
    n_ubatch: int = 512
    n_gpu_layers: int = -1
    verbose: bool = False


@dataclass(frozen=True)
class TaskConfig:
    prompt_file: Path
    hop1_template_file: Path
    hop2_template_file: Path
    target_eog: str
    max_hop1_tokens: int = 256
    add_bos: bool = False


@dataclass(frozen=True)
class RidgeConfig:
    enabled: bool = False
    minimum_observations: int = 256
    regularization: float = 0.01
    exploration_fraction: float = 0.25


@dataclass(frozen=True)
class SearchConfig:
    output_dir: Path
    steps: int = 20
    candidate_budget: int = 512
    top_k: int = 256
    rank_temperature: float = 0.1
    full_recheck_count: int = 8
    vocab_chunk_size: int = 8192
    seed: int = 42
    stop_when_exact: bool = True
    mutable_positions: tuple[int, ...] = ()
    ridge: RidgeConfig = field(default_factory=RidgeConfig)


@dataclass(frozen=True)
class Config:
    source: Path
    model: ModelConfig
    task: TaskConfig
    search: SearchConfig


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing [{name}] table")
    return value


def load_config(path: str | Path) -> Config:
    source = Path(path).expanduser().resolve()
    base = source.parent
    with source.open("rb") as handle:
        raw = tomllib.load(handle)

    model_raw = _table(raw, "model")
    task_raw = _table(raw, "task")
    search_raw = _table(raw, "search")

    model = ModelConfig(
        gguf_path=_resolve_path(base, str(model_raw["gguf_path"])),
        hf_model=str(model_raw["hf_model"]),
        model_class=str(model_raw.get("model_class", "AutoModelForCausalLM")),
        dtype=str(model_raw.get("dtype", "bfloat16")),
        device_map=str(model_raw.get("device_map", "auto")),
        attn_implementation=(
            str(model_raw["attn_implementation"])
            if model_raw.get("attn_implementation")
            else None
        ),
        trust_remote_code=bool(model_raw.get("trust_remote_code", False)),
        n_ctx=int(model_raw.get("n_ctx", 8192)),
        n_batch=int(model_raw.get("n_batch", 512)),
        n_ubatch=int(model_raw.get("n_ubatch", 512)),
        n_gpu_layers=int(model_raw.get("n_gpu_layers", -1)),
        verbose=bool(model_raw.get("verbose", False)),
    )
    task = TaskConfig(
        prompt_file=_resolve_path(base, str(task_raw["prompt_file"])),
        hop1_template_file=_resolve_path(
            base, str(task_raw["hop1_template_file"])
        ),
        hop2_template_file=_resolve_path(
            base, str(task_raw["hop2_template_file"])
        ),
        target_eog=str(task_raw["target_eog"]),
        max_hop1_tokens=int(task_raw.get("max_hop1_tokens", 256)),
        add_bos=bool(task_raw.get("add_bos", False)),
    )
    mutable = tuple(int(value) for value in search_raw.get("mutable_positions", []))
    ridge_raw = search_raw.get("ridge", {})
    if not isinstance(ridge_raw, dict):
        raise ValueError("[search.ridge] must be a table")
    ridge = RidgeConfig(
        enabled=bool(ridge_raw.get("enabled", False)),
        minimum_observations=int(ridge_raw.get("minimum_observations", 256)),
        regularization=float(ridge_raw.get("regularization", 0.01)),
        exploration_fraction=float(ridge_raw.get("exploration_fraction", 0.25)),
    )
    search = SearchConfig(
        output_dir=_resolve_path(base, str(search_raw.get("output_dir", "runs/demo"))),
        steps=int(search_raw.get("steps", 20)),
        candidate_budget=int(search_raw.get("candidate_budget", 512)),
        top_k=int(search_raw.get("top_k", 256)),
        rank_temperature=float(search_raw.get("rank_temperature", 0.1)),
        full_recheck_count=int(search_raw.get("full_recheck_count", 8)),
        vocab_chunk_size=int(search_raw.get("vocab_chunk_size", 8192)),
        seed=int(search_raw.get("seed", 42)),
        stop_when_exact=bool(search_raw.get("stop_when_exact", True)),
        mutable_positions=mutable,
        ridge=ridge,
    )
    config = Config(source=source, model=model, task=task, search=search)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    if config.model.n_ctx <= 0 or config.model.n_batch <= 0:
        raise ValueError("n_ctx and n_batch must be positive")
    if config.model.n_ubatch <= 0:
        raise ValueError("n_ubatch must be positive")
    if config.task.max_hop1_tokens <= 0:
        raise ValueError("max_hop1_tokens must be positive")
    if config.search.steps <= 0 or config.search.candidate_budget <= 0:
        raise ValueError("steps and candidate_budget must be positive")
    if config.search.top_k <= 0 or config.search.full_recheck_count <= 0:
        raise ValueError("top_k and full_recheck_count must be positive")
    if config.search.rank_temperature <= 0.0:
        raise ValueError("rank_temperature must be positive")
    if config.search.vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    if config.search.ridge.minimum_observations <= 0:
        raise ValueError("ridge minimum_observations must be positive")
    if config.search.ridge.regularization <= 0.0:
        raise ValueError("ridge regularization must be positive")
    if not 0.0 <= config.search.ridge.exploration_fraction < 1.0:
        raise ValueError("ridge exploration_fraction must be in [0, 1)")
