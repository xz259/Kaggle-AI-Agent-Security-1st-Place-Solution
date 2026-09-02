"""Numerically stable token and ordered-sequence objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TokenScore:
    target_id: int
    target_logit: float
    competitor_id: int
    competitor_logit: float
    greedy_id: int
    nll: float
    margin: float

    @property
    def exact(self) -> bool:
        return self.greedy_id == self.target_id and self.margin > 0.0

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "target_id": self.target_id,
            "target_logit": self.target_logit,
            "competitor_id": self.competitor_id,
            "competitor_logit": self.competitor_logit,
            "greedy_id": self.greedy_id,
            "nll": self.nll,
            "margin": self.margin,
            "exact": self.exact,
        }


@dataclass(frozen=True)
class SequenceScore:
    tokens: tuple[TokenScore, ...]

    @property
    def exact(self) -> bool:
        return bool(self.tokens) and all(token.exact for token in self.tokens)

    @property
    def mean_nll(self) -> float:
        return float(np.mean([token.nll for token in self.tokens]))

    @property
    def min_margin(self) -> float:
        return min(token.margin for token in self.tokens)

    def as_dict(self) -> dict[str, object]:
        return {
            "exact": self.exact,
            "mean_nll": self.mean_nll,
            "min_margin": self.min_margin,
            "tokens": [token.as_dict() for token in self.tokens],
        }


def score_logits(logits: Sequence[float] | np.ndarray, target_id: int) -> TokenScore:
    values = np.asarray(logits, dtype=np.float64)
    target = int(target_id)
    if values.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if not bool(np.isfinite(values).all()):
        raise ValueError("logits contain non-finite values")
    if target < 0 or target >= values.size:
        raise IndexError(f"target token {target} outside vocabulary")

    masked = values.copy()
    masked[target] = -np.inf
    competitor = int(np.argmax(masked))
    greedy = int(np.argmax(values))
    maximum = float(np.max(values))
    log_z = maximum + float(np.log(np.exp(values - maximum).sum()))
    target_logit = float(values[target])
    competitor_logit = float(values[competitor])
    return TokenScore(
        target_id=target,
        target_logit=target_logit,
        competitor_id=competitor,
        competitor_logit=competitor_logit,
        greedy_id=greedy,
        nll=log_z - target_logit,
        margin=target_logit - competitor_logit,
    )
