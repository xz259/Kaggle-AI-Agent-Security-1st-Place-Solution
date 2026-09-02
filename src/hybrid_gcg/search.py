"""Radius-1 hybrid GCG loop with exact hop-1 preservation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .config import Config
from .hotflip import (
    HotFlipProposer,
    Proposal,
    Rankings,
    build_radius1_panel,
    prompt_hash,
)
from .llama_backend import LlamaBackend
from .objectives import SequenceScore, TokenScore
from .templates import TextTemplate, TokenBoundaryError
from .trajectory import Trajectory


class Proposer(Protocol):
    def rank(
        self,
        prompt_ids: Sequence[int],
        trajectory: Trajectory,
        mutable_positions: Sequence[int],
    ) -> Rankings: ...


@dataclass(frozen=True)
class Candidate:
    proposal: Proposal
    prompt_ids: tuple[int, ...]
    prompt_text: str
    hop1_context_ids: tuple[int, ...]
    hop2_context_ids: tuple[int, ...]


@dataclass(frozen=True)
class ValidatedCandidate:
    candidate: Candidate
    cached_hop2: TokenScore
    full_hop2: TokenScore
    hop1: SequenceScore


def better_hop2(
    candidate: TokenScore,
    incumbent: TokenScore,
    epsilon: float = 1e-8,
) -> bool:
    return candidate.margin > incumbent.margin + epsilon


class SearchRunner:
    def __init__(
        self,
        config: Config,
        trajectory: Trajectory,
        backend: LlamaBackend,
        proposer: Proposer | None = None,
    ) -> None:
        self.config = config
        self.trajectory = trajectory
        self.backend = backend
        self._validate_provenance()
        self.hop1_template = TextTemplate.load(config.task.hop1_template_file)
        self.hop2_template = TextTemplate.load(config.task.hop2_template_file)
        self.mutable_positions = self._mutable_positions()
        self.proposer = proposer or HotFlipProposer(
            config.model,
            trajectory,
            top_k=config.search.top_k,
            vocab_chunk_size=config.search.vocab_chunk_size,
        )
        self.rng = np.random.default_rng(config.search.seed)
        self.output_dir = config.search.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.events_path = self.output_dir / "events.jsonl"
        self.result_path = self.output_dir / "result.json"
        self.fingerprint = self._fingerprint()

    def _validate_provenance(self) -> None:
        if self.trajectory.llama_cpp_version != self.backend.version:
            raise RuntimeError(
                "llama.cpp version differs from the bootstrap baseline: "
                f"{self.trajectory.llama_cpp_version!r} != {self.backend.version!r}"
            )
        if self.trajectory.gguf_name != self.config.model.gguf_path.name:
            raise RuntimeError(
                "GGUF filename differs from the bootstrap baseline: "
                f"{self.trajectory.gguf_name!r} != "
                f"{self.config.model.gguf_path.name!r}"
            )
        size = self.config.model.gguf_path.stat().st_size
        if self.trajectory.gguf_size_bytes != size:
            raise RuntimeError(
                "GGUF file size differs from the bootstrap baseline: "
                f"{self.trajectory.gguf_size_bytes} != {size}"
            )
        if self.trajectory.gguf_vocab_size != self.backend.n_vocab:
            raise RuntimeError(
                "GGUF vocabulary size differs from the bootstrap baseline: "
                f"{self.trajectory.gguf_vocab_size} != {self.backend.n_vocab}"
            )

    def _mutable_positions(self) -> tuple[int, ...]:
        length = len(self.trajectory.prompt_token_ids)
        configured = self.config.search.mutable_positions
        positions = configured or tuple(range(length))
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("mutable_positions must be non-empty and unique")
        if min(positions) < 0 or max(positions) >= length:
            raise ValueError("mutable_positions contains an out-of-range index")
        return tuple(positions)

    def _fingerprint(self) -> str:
        payload = {
            "trajectory": self.trajectory.to_dict(),
            "hf_model": self.config.model.hf_model,
            "top_k": self.config.search.top_k,
            "candidate_budget": self.config.search.candidate_budget,
            "rank_temperature": self.config.search.rank_temperature,
            "full_recheck_count": self.config.search.full_recheck_count,
            "vocab_chunk_size": self.config.search.vocab_chunk_size,
            "seed": self.config.search.seed,
            "mutable_positions": self.mutable_positions,
            "hop1_template": self.hop1_template.text,
            "hop2_template": self.hop2_template.text,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _event(self, payload: dict[str, Any]) -> None:
        row = {"time": time.time(), **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def _save_checkpoint(
        self,
        *,
        next_step: int,
        current_ids: Sequence[int],
        visited: set[str],
        history: list[dict[str, Any]],
    ) -> None:
        payload = {
            "schema_version": 1,
            "fingerprint": self.fingerprint,
            "next_step": int(next_step),
            "current_prompt_ids": list(map(int, current_ids)),
            "visited_prompt_hashes": sorted(visited),
            "rng_state": self.rng.bit_generator.state,
            "history": history,
        }
        self.checkpoint_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _load_checkpoint(
        self,
    ) -> tuple[int, tuple[int, ...], set[str], list[dict[str, Any]]]:
        if not self.checkpoint_path.exists():
            return 0, self.trajectory.prompt_token_ids, set(), []
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != self.fingerprint:
            raise RuntimeError(
                "Checkpoint does not match this trajectory/configuration"
            )
        self.rng.bit_generator.state = payload["rng_state"]
        return (
            int(payload["next_step"]),
            tuple(map(int, payload["current_prompt_ids"])),
            set(map(str, payload["visited_prompt_hashes"])),
            list(payload.get("history", [])),
        )

    def _candidate(
        self,
        current_ids: Sequence[int],
        proposal: Proposal,
    ) -> Candidate | None:
        tokens = list(map(int, current_ids))
        tokens[proposal.position] = proposal.replacement_id
        candidate_ids = tuple(tokens)
        try:
            text = self.backend.detokenize(candidate_ids, special=True)
            if self.backend.tokenize(text, False) != candidate_ids:
                return None
            hop1 = self.hop1_template.render_token_aligned(
                text,
                self.backend.tokenize,
                add_bos=self.config.task.add_bos,
            )
            hop2 = self.hop2_template.render_token_aligned(
                text,
                self.backend.tokenize,
                add_bos=self.config.task.add_bos,
                hop1=self.trajectory.hop1_text,
                hop1_visible=self.trajectory.hop1_visible_text,
            )
        except (UnicodeDecodeError, TokenBoundaryError):
            return None
        if tuple(hop1.token_ids[slice(*hop1.prompt_span)]) != candidate_ids:
            return None
        if tuple(hop2.token_ids[slice(*hop2.prompt_span)]) != candidate_ids:
            return None
        return Candidate(
            proposal=proposal,
            prompt_ids=candidate_ids,
            prompt_text=text,
            hop1_context_ids=hop1.token_ids,
            hop2_context_ids=hop2.token_ids,
        )

    def _validate_hop1(self, candidate: Candidate) -> SequenceScore | None:
        score = self.backend.score_sequence(
            candidate.hop1_context_ids,
            self.trajectory.hop1_output_ids,
        )
        if not score.exact:
            return None
        greedy = self.backend.greedy_decode(
            candidate.hop1_context_ids,
            max_tokens=len(self.trajectory.hop1_output_ids),
        )
        if not greedy.stopped_on_eog:
            return None
        if greedy.token_ids != self.trajectory.hop1_output_ids:
            return None
        return score

    def _assert_incumbent_hop1(self, current_ids: Sequence[int]) -> Candidate:
        no_op = Proposal(
            position=self.mutable_positions[0],
            replacement_id=int(current_ids[self.mutable_positions[0]]),
            gradient_rank=-1,
            predicted_delta=0.0,
        )
        candidate = self._candidate(current_ids, no_op)
        if candidate is None or self._validate_hop1(candidate) is None:
            raise RuntimeError(
                "The incumbent no longer reproduces the bootstrapped hop 1 exactly"
            )
        return candidate

    def run(self) -> dict[str, Any]:
        start_step, current_ids, visited, history = self._load_checkpoint()
        incumbent_candidate = self._assert_incumbent_hop1(current_ids)
        incumbent = self.backend.score_next(
            incumbent_candidate.hop2_context_ids,
            self.trajectory.target_eog_id,
        )
        self._event(
            {
                "event": "start",
                "step": start_step,
                "hop2": incumbent.as_dict(),
                "mutable_count": len(self.mutable_positions),
            }
        )
        if incumbent.exact and self.config.search.stop_when_exact:
            return self._finish(current_ids, incumbent, history, "already_exact")

        for step in range(start_step, self.config.search.steps):
            rankings = self.proposer.rank(
                current_ids, self.trajectory, self.mutable_positions
            )
            panel = build_radius1_panel(
                current_ids,
                rankings,
                budget=self.config.search.candidate_budget,
                temperature=self.config.search.rank_temperature,
                rng=self.rng,
                visited=visited,
            )
            if not panel:
                self._event(
                    {
                        "event": "proposal_space_exhausted",
                        "step": step,
                        "gradient_loss": rankings.loss,
                    }
                )
                return self._finish(
                    current_ids,
                    incumbent,
                    history,
                    "proposal_space_exhausted",
                )
            candidates: list[Candidate] = []
            for proposal in panel:
                tokens = list(current_ids)
                tokens[proposal.position] = proposal.replacement_id
                visited.add(prompt_hash(tokens))
                candidate = self._candidate(current_ids, proposal)
                if candidate is not None:
                    candidates.append(candidate)
            if not candidates:
                event = {
                    "event": "panel",
                    "step": step,
                    "gradient_loss": rankings.loss,
                    "gradient_competitor_id": rankings.competitor_id,
                    "proposals": len(panel),
                    "tokenizer_valid": 0,
                    "full_rechecks": 0,
                    "hop1_valid_improvements": 0,
                    "accepted": False,
                    "incumbent_hop2": incumbent.as_dict(),
                }
                history.append(event)
                self._event(event)
                self._save_checkpoint(
                    next_step=step + 1,
                    current_ids=current_ids,
                    visited=visited,
                    history=history,
                )
                continue

            cached_scores = self.backend.score_next_panel(
                [candidate.hop2_context_ids for candidate in candidates],
                common_prefix_length=self.trajectory.hop2_prompt_span[0],
                target_id=self.trajectory.target_eog_id,
            )
            ranked = sorted(
                zip(candidates, cached_scores),
                key=lambda row: row[1].margin,
                reverse=True,
            )
            shortlist = ranked[: self.config.search.full_recheck_count]

            valid: list[ValidatedCandidate] = []
            for candidate, cached in shortlist:
                full = self.backend.score_next(
                    candidate.hop2_context_ids,
                    self.trajectory.target_eog_id,
                )
                if not better_hop2(full, incumbent):
                    continue
                hop1 = self._validate_hop1(candidate)
                if hop1 is None:
                    continue
                valid.append(
                    ValidatedCandidate(
                        candidate=candidate,
                        cached_hop2=cached,
                        full_hop2=full,
                        hop1=hop1,
                    )
                )

            accepted = max(valid, key=lambda row: row.full_hop2.margin, default=None)
            event: dict[str, Any] = {
                "event": "panel",
                "step": step,
                "gradient_loss": rankings.loss,
                "gradient_competitor_id": rankings.competitor_id,
                "proposals": len(panel),
                "tokenizer_valid": len(candidates),
                "full_rechecks": len(shortlist),
                "hop1_valid_improvements": len(valid),
                "accepted": accepted is not None,
                "incumbent_hop2": incumbent.as_dict(),
            }
            if accepted is not None:
                old_margin = incumbent.margin
                current_ids = accepted.candidate.prompt_ids
                incumbent = accepted.full_hop2
                event["mutation"] = {
                    "position": accepted.candidate.proposal.position,
                    "replacement_id": accepted.candidate.proposal.replacement_id,
                    "replacement_piece": self.backend.token_piece(
                        accepted.candidate.proposal.replacement_id
                    ),
                    "gradient_rank": accepted.candidate.proposal.gradient_rank,
                    "predicted_delta": accepted.candidate.proposal.predicted_delta,
                }
                event["margin_gain"] = incumbent.margin - old_margin
                event["accepted_hop2"] = incumbent.as_dict()
                event["hop1"] = accepted.hop1.as_dict()
                event["prompt"] = accepted.candidate.prompt_text
            history.append(event)
            self._event(event)
            self._save_checkpoint(
                next_step=step + 1,
                current_ids=current_ids,
                visited=visited,
                history=history,
            )
            if incumbent.exact and self.config.search.stop_when_exact:
                return self._finish(current_ids, incumbent, history, "target_exact")

        return self._finish(current_ids, incumbent, history, "step_limit")

    def _finish(
        self,
        current_ids: Sequence[int],
        incumbent: TokenScore,
        history: list[dict[str, Any]],
        reason: str,
    ) -> dict[str, Any]:
        final_candidate = self._assert_incumbent_hop1(current_ids)
        hop2_greedy = self.backend.greedy_decode(
            final_candidate.hop2_context_ids,
            max_tokens=1,
        )
        solved = (
            hop2_greedy.stopped_on_eog
            and hop2_greedy.token_ids == (self.trajectory.target_eog_id,)
        )
        result = {
            "schema_version": 1,
            "reason": reason,
            "solved": solved,
            "prompt": final_candidate.prompt_text,
            "prompt_token_ids": list(map(int, current_ids)),
            "hop1_preserved": True,
            "hop2": incumbent.as_dict(),
            "hop2_greedy_ids": list(hop2_greedy.token_ids),
            "accepted_steps": sum(bool(row.get("accepted")) for row in history),
            "panels": len(history),
        }
        self.result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._event({"event": "finish", **result})
        return result
