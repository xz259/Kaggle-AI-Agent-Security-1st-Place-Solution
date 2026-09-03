"""GGUF-calibrated ridge reranking for radius-1 HotFlip proposals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .hotflip import (
    Proposal,
    Rankings,
    build_radius1_panel,
    prompt_hash,
    rank_probabilities,
)


def _design_matrix(
    positions: np.ndarray,
    ranks: np.ndarray,
    deltas: np.ndarray,
    *,
    active_positions: Sequence[int],
    delta_mean: float,
    delta_scale: float,
    top_k: int,
) -> np.ndarray:
    """Build the local feature matrix used in the competition search.

    The HotFlip delta is only one feature. Coordinate intercepts and
    coordinate-specific slopes let exact GGUF observations correct systematic
    transfer error without treating token IDs as ordinal values.
    """
    position_to_local = {
        int(position): index for index, position in enumerate(active_positions)
    }
    local = np.asarray(
        [position_to_local[int(position)] for position in positions],
        dtype=np.int64,
    )
    scaled_delta = (
        np.asarray(deltas, dtype=np.float64) - float(delta_mean)
    ) / float(delta_scale)
    rank_fraction = np.asarray(ranks, dtype=np.float64) / max(1, top_k - 1)
    count = len(local)
    n_positions = len(active_positions)
    design = np.zeros((count, 5 + 2 * n_positions), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1] = scaled_delta
    design[:, 2] = rank_fraction
    design[:, 3] = scaled_delta * rank_fraction
    design[:, 4] = scaled_delta * scaled_delta
    rows = np.arange(count)
    design[rows, 5 + local] = 1.0
    design[rows, 5 + n_positions + local] = scaled_delta
    return design


def _ridge_predictions(
    train_design: np.ndarray,
    targets: np.ndarray,
    predict_design: np.ndarray,
    *,
    regularization: float,
) -> tuple[np.ndarray, dict[str, float | None]]:
    if regularization <= 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    penalty = np.eye(train_design.shape[1], dtype=np.float64) * regularization
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        train_design.T @ train_design + penalty,
        train_design.T @ np.asarray(targets, dtype=np.float64),
    )
    fitted = train_design @ coefficients
    residual = fitted - targets
    correlation: float | None
    if np.std(fitted) == 0.0 or np.std(targets) == 0.0:
        correlation = None
    else:
        correlation = float(np.corrcoef(fitted, targets)[0, 1])
    diagnostics = {
        "train_rmse": float(np.sqrt(np.mean(residual * residual))),
        "train_correlation": correlation,
    }
    return predict_design @ coefficients, diagnostics


def _weighted_pair_order(
    n_positions: int,
    top_k: int,
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sample position/rank pairs without replacement using Gumbel keys."""
    pairs: list[tuple[int, int]] = []
    keys: list[float] = []
    for position_index in range(n_positions):
        for rank in range(1, top_k):
            pairs.append((position_index, rank))
            keys.append(
                math.log(float(probabilities[rank])) + float(rng.gumbel())
            )
    order = np.argsort(np.asarray(keys))[::-1]
    return [pairs[int(index)] for index in order]


def build_ridge_panel(
    current_ids: Sequence[int],
    rankings: Rankings,
    *,
    budget: int,
    temperature: float,
    rng: np.random.Generator,
    visited: set[str],
    observations: Sequence[Mapping[str, Any]],
    exploration_fraction: float,
    regularization: float,
    minimum_observations: int,
    cover_coordinates: bool = True,
) -> tuple[list[Proposal], dict[str, Any]]:
    """Rerank unseen HotFlip proposals using local exact-GGUF labels.

    Two ridge models predict GGUF NLL change and GGUF margin gain. Their orders
    are interleaved after a reserved Faster-GCG exploration allocation. The
    returned proposals still require the caller's normal GGUF and greedy gates.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not 0.0 <= exploration_fraction < 1.0:
        raise ValueError("exploration_fraction must be in [0, 1)")
    if minimum_observations <= 0:
        raise ValueError("minimum_observations must be positive")
    positions = tuple(map(int, rankings.positions))
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("ranked positions must be non-empty and unique")
    if rankings.token_ids.shape != rankings.predicted_deltas.shape:
        raise ValueError("ranked ID/delta shape mismatch")
    if rankings.token_ids.shape[0] != len(positions):
        raise ValueError("ranked position count does not match proposal rows")
    top_k = int(rankings.token_ids.shape[1])
    if top_k <= 0:
        raise ValueError("rankings must contain at least one token per position")

    active = set(positions)
    usable = [
        row
        for row in observations
        if int(row["position"]) in active
        and math.isfinite(float(row["predicted_delta"]))
        and math.isfinite(float(row["loss_delta"]))
        and math.isfinite(float(row["margin_gain"]))
    ]
    if len(usable) < minimum_observations:
        panel = build_radius1_panel(
            current_ids,
            rankings,
            budget=budget,
            temperature=temperature,
            rng=rng,
            visited=visited,
            source="gradient_fallback",
            cover_coordinates=cover_coordinates,
        )
        return panel, {
            "strategy": "gradient_fallback_insufficient_gguf_observations",
            "observation_count": len(usable),
            "minimum_observations": minimum_observations,
            "panel_sources": {"gradient_fallback": len(panel)},
        }

    universe_positions = np.repeat(np.asarray(positions, dtype=np.int64), top_k)
    universe_ranks = np.tile(np.arange(top_k, dtype=np.int64), len(positions))
    universe_deltas = rankings.predicted_deltas.reshape(-1).astype(np.float64)
    delta_mean = float(np.mean(universe_deltas))
    delta_scale = float(np.std(universe_deltas))
    if not math.isfinite(delta_scale) or delta_scale < 1e-12:
        delta_scale = 1.0

    train_positions = np.asarray(
        [int(row["position"]) for row in usable], dtype=np.int64
    )
    train_ranks = np.asarray(
        [int(row["gradient_rank"]) for row in usable], dtype=np.int64
    )
    train_deltas = np.asarray(
        [float(row["predicted_delta"]) for row in usable], dtype=np.float64
    )
    train_design = _design_matrix(
        train_positions,
        train_ranks,
        train_deltas,
        active_positions=positions,
        delta_mean=delta_mean,
        delta_scale=delta_scale,
        top_k=top_k,
    )
    predict_design = _design_matrix(
        universe_positions,
        universe_ranks,
        universe_deltas,
        active_positions=positions,
        delta_mean=delta_mean,
        delta_scale=delta_scale,
        top_k=top_k,
    )
    predicted_loss, loss_fit = _ridge_predictions(
        train_design,
        np.asarray([float(row["loss_delta"]) for row in usable]),
        predict_design,
        regularization=regularization,
    )
    predicted_margin, margin_fit = _ridge_predictions(
        train_design,
        np.asarray([float(row["margin_gain"]) for row in usable]),
        predict_design,
        regularization=regularization,
    )

    probabilities = rank_probabilities(top_k, temperature)
    weighted_pairs = _weighted_pair_order(
        len(positions), top_k, probabilities, rng
    )
    loss_order = np.argsort(predicted_loss, kind="stable")
    margin_order = np.argsort(-predicted_margin, kind="stable")
    panel: list[Proposal] = []
    panel_hashes: set[str] = set()
    counts: dict[str, Any] = {
        "strategy": "gguf_ridge_calibrated",
        "observation_count": len(usable),
        "minimum_observations": minimum_observations,
        "regularization": regularization,
        "exploration_fraction": exploration_fraction,
        "loss_fit": loss_fit,
        "margin_fit": margin_fit,
        "pairs_considered": 0,
        "no_op": 0,
        "previously_visited": 0,
        "duplicate_in_panel": 0,
    }

    def consider(position_index: int, rank: int, source: str) -> bool:
        counts["pairs_considered"] += 1
        position = positions[position_index]
        replacement = int(rankings.token_ids[position_index, rank])
        if replacement == int(current_ids[position]):
            counts["no_op"] += 1
            return False
        candidate = list(map(int, current_ids))
        candidate[position] = replacement
        key = prompt_hash(candidate)
        if key in visited:
            counts["previously_visited"] += 1
            return False
        if key in panel_hashes:
            counts["duplicate_in_panel"] += 1
            return False
        panel_hashes.add(key)
        panel.append(
            Proposal(
                position=position,
                replacement_id=replacement,
                gradient_rank=rank,
                predicted_delta=float(
                    rankings.predicted_deltas[position_index, rank]
                ),
                source=source,
            )
        )
        return True

    if cover_coordinates:
        for position_index in range(len(positions)):
            for rank in range(top_k):
                if consider(position_index, rank, "ridge_coverage"):
                    break
            if len(panel) >= budget:
                break

    exploration_target = max(
        len(panel), min(budget, int(round(budget * exploration_fraction)))
    )
    for position_index, rank in weighted_pairs:
        if len(panel) >= exploration_target:
            break
        consider(position_index, rank, "ridge_exploration")

    loss_cursor = 0
    margin_cursor = 0
    while len(panel) < budget and (
        loss_cursor < len(loss_order) or margin_cursor < len(margin_order)
    ):
        if loss_cursor < len(loss_order):
            index = int(loss_order[loss_cursor])
            loss_cursor += 1
            position_index, rank = divmod(index, top_k)
            consider(position_index, rank, "ridge_loss")
        if len(panel) >= budget:
            break
        if margin_cursor < len(margin_order):
            index = int(margin_order[margin_cursor])
            margin_cursor += 1
            position_index, rank = divmod(index, top_k)
            consider(position_index, rank, "ridge_margin")

    if len(panel) < budget:
        for position_index, rank in weighted_pairs:
            if len(panel) >= budget:
                break
            consider(position_index, rank, "ridge_fill")

    counts["accepted"] = len(panel)
    counts["panel_sources"] = {
        source: sum(row.source == source for row in panel)
        for source in sorted({row.source for row in panel})
    }
    return panel, counts
